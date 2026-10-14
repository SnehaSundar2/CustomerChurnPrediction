"""The training workflow, step by step.

    raw data → clean → split (train/test, same churn share) → compare models by cross-validation (train only)
             → tune the best → decide whether to calibrate its probabilities → out-of-fold probabilities on
             the training set → choose the contact threshold from the campaign economics
             → evaluate ONCE on the untouched test set → explain → save

The test set is used exactly once, for the final report; every choice (which model, which settings, whether to
calibrate, which threshold) is made with cross-validation on the training customers.
"""

import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import __version__
from . import evaluation as E
from . import models as M
from . import preprocessing as P
from . import schema as S
from .exceptions import DataError
from .validators import validate_economics

log = logging.getLogger("churnpredict")


@dataclass
class TrainingResult:
    pipeline: object
    metadata: dict
    comparison: list
    cleaning: P.CleaningReport
    test_frame: pd.DataFrame          # features + actual / probability / predicted for every test customer
    importance: list
    reference: pd.DataFrame           # cleaned TRAINING customers (used to find similar customers)
    segments: dict = field(default_factory=dict)


def train(df_raw, seed=42, fast=False, tune=True, test_size=0.2, cv=5, progress=None, source="dataset",
          value=S.CUSTOMER_VALUE, cost=S.OFFER_COST, save_rate=S.SAVE_RATE):
    """Run the whole workflow. `progress(message)` is called as each stage starts."""
    say = progress or (lambda message: None)
    started = datetime.now()
    value, cost, save_rate = validate_economics(value, cost, save_rate)

    say("Cleaning the data ...")
    df, report = P.clean(df_raw)
    X, y = P.split_xy(df)

    say("Splitting into training and test sets ...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    if y_train.sum() < cv or (len(y_train) - y_train.sum()) < cv:
        raise DataError(f"Too few customers of one kind to cross-validate ({int(y_train.sum())} churners in training).")

    comparison = M.compare(X_train, y_train, cv=cv, seed=seed, fast=fast, progress=say)
    best = M.best_candidate(comparison)
    chosen, params = best["model"], {}

    tuned_note = ""
    if tune and chosen in M.TUNING_GRIDS:
        _, params = M.tune(chosen, X_train, y_train, seed=seed, fast=fast, progress=say)
        if params:
            say("Checking the tuned settings ...")
            scores = cross_validate(M.pipeline_with_params(chosen, params, seed, fast), X_train, y_train,
                                    cv=StratifiedKFold(cv, shuffle=True, random_state=seed), scoring=M.SCORING)
            tuned = M._summarise(f"{chosen} (tuned)", scores, 0.0)
            if tuned["roc_auc"] > best["roc_auc"]:
                comparison = sorted(comparison + [tuned], key=lambda r: (r["roc_auc"], r["average_precision"]), reverse=True)
                best, tuned_note = tuned, "tuned"
            else:
                params = {}
                tuned_note = "tuning did not improve cross-validation, kept defaults"

    say("Checking whether the probabilities need calibrating ...")
    base = (M.pipeline_with_params(chosen, params, seed, fast) if params
            else M.build_pipeline(M.candidates(seed, fast)[chosen]))
    final, oof, calibration = M.choose_calibration(base, X_train, y_train, seed=seed, cv=cv)

    say("Training the final model ...")
    final.fit(X_train, y_train)

    say("Choosing the contact threshold from the campaign economics (training data only) ...")
    economics = E.choose_contact_threshold(y_train, oof, value, cost, save_rate)
    t = economics["contact"]

    say("Evaluating on the untouched test set ...")
    test_p = final.predict_proba(X_test)[:, 1]
    train_p = final.predict_proba(X_train)[:, 1]
    test_metrics = {**E.classification_metrics(y_test, test_p, t), **E.ranking_metrics(y_test, test_p)}
    test_metrics["default_threshold"] = E.classification_metrics(y_test, test_p, 0.5)
    test_metrics["majority_accuracy"] = float(100 * max(y_test.mean(), 1 - y_test.mean()))
    test_metrics["churn_rate"] = float(100 * y_test.mean())
    train_metrics = {**E.classification_metrics(y_train, train_p, t), **E.ranking_metrics(y_train, train_p)}
    baseline_row = next(r for r in comparison if r["model"] == M.BASELINE)
    policies = E.policy_table(y_test, test_p, t, value, cost, save_rate)

    say("Measuring feature importance ...")
    importance = E.feature_importance(final, X_test, y_test, seed=seed, repeats=3 if fast else 8)

    test_frame = X_test.copy()
    test_frame["actual"] = y_test.to_numpy()
    test_frame["probability"] = test_p
    test_frame["predicted"] = (test_p >= t).astype(int)
    segments = {"contract": E.segment_rows(test_frame, "contract"), "internet": E.segment_rows(test_frame, "internet_service"),
                "payment": E.segment_rows(test_frame, "payment_method")}

    reference = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    metadata = {
        "project_version": __version__,
        "trained_at": started.isoformat(timespec="seconds"),
        "training_seconds": round((datetime.now() - started).total_seconds(), 1),
        "source": source,
        "seed": seed,
        "model": chosen,
        "model_note": tuned_note,
        "model_params": {k: (v if v is None or isinstance(v, (int, float, str)) else str(v)) for k, v in params.items()},
        "calibration": calibration,
        "rows": {"raw": report.rows_in, "clean": report.rows_out, "train": int(len(X_train)), "test": int(len(X_test))},
        "class_counts": report.class_counts,
        "metrics": {"test": test_metrics, "train": train_metrics, "cv": best, "baseline_cv": baseline_row},
        "economics": economics,
        "thresholds": {"contact": t, "watch": t / 2},
        "policies": policies,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__, "platform": sys.platform},
        "training_ranges": {c: [float(df[c].min()), float(df[c].max())] for c in S.NUMERIC_FEATURES},
        "typical": {**{c: float(df[c].median()) for c in S.NUMERIC_FEATURES},
                    **{c: int(round(df[c].median())) for c in S.BINARY_FEATURES},
                    **{c: str(df[c].mode().iat[0]) for c in S.CATEGORICAL_FEATURES}},
        "importance": importance,
        "comparison": comparison,
        "cleaning": report.to_dict(),
        "segments": segments,
        "lift": E.lift_table(y_test, test_p),
        "gains": E.gains_points(y_test, test_p),
        "calibration_table": E.calibration_table(y_test, test_p),
        "profit_curve": E.profit_curve(y_test, test_p, value, cost, save_rate),
        "ceiling": E.roc_curve_ceiling(df) if str(source).startswith("synthetic") else None,
    }
    log.info("Trained %s: test AUC=%.3f profit/1000=%.0f", chosen, test_metrics["roc_auc"], policies[2]["profit_per_1000"])
    return TrainingResult(final, metadata, comparison, report, test_frame, importance, reference, segments)
