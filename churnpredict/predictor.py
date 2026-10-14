"""Scoring customers: churn probability, risk tier, offer decision, the factors behind it, retention actions,
what-if analysis, similar customers and batch files."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import persistence
from . import schema as S
from .exceptions import DataError, NotTrainedError, ValidationError
from .preprocessing import normalise_headers
from .validators import REQUIRED_INPUT, parse_features

MAX_BATCH_ROWS = 5000
MIN_EFFECT = 0.005       # a factor or action must move the churn chance by at least half a point to be shown


@dataclass
class Prediction:
    features: dict
    assumed: dict
    probability: float
    tier: str
    contact: bool
    expected_gain: float          # expected profit of making a retention offer to this customer (can be negative)
    warnings: list = field(default_factory=list)
    factors: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    similar: list = field(default_factory=list)

    def to_dict(self):
        return {"churn_probability": round(self.probability, 4), "risk_tier": self.tier, "make_offer": self.contact,
                "expected_gain": round(self.expected_gain, 2), "features": self.features,
                "assumed_defaults": self.assumed, "warnings": self.warnings, "factors": self.factors,
                "retention_actions": self.actions, "similar_customers": self.similar}


def _frame(rows):
    """DataFrame in the exact column order the pipeline was trained on."""
    return pd.DataFrame(rows)[S.RAW_FEATURES]


class Predictor:
    def __init__(self, loaded):
        self.model = loaded.pipeline
        self.meta = loaded.metadata
        self.reference = loaded.reference
        self.load_warnings = list(loaded.warnings)
        self.threshold = self.meta["thresholds"]["contact"]
        self.economics = self.meta["economics"]
        self._typical = self.meta["typical"]
        self._ranges = self.meta["training_ranges"]

    @classmethod
    def from_directory(cls, directory):
        return cls(persistence.load(directory))

    # ------------------------------------------------------------------ core
    def _proba(self, rows):
        return np.asarray(self.model.predict_proba(_frame(rows))[:, 1], float)

    def gain(self, probability):
        """Expected profit of an offer: the chance the customer is saved (× value) minus the cost of the offer."""
        e = self.economics
        return float(probability * e["save_rate"] * e["value"] - e["cost"])

    def _range_warnings(self, f):
        out = []
        for name in S.NUMERIC_FEATURES:
            low, high = self._ranges[name]
            if f[name] < low or f[name] > high:
                out.append(f"{S.LABELS[name]} {f[name]:g} is outside the range the model learned "
                           f"({low:g}–{high:g}); the estimate is less reliable.")
        return out

    def predict(self, raw, explain=True, similar=True):
        features, assumed = parse_features(raw)
        p = float(self._proba([features])[0])
        result = Prediction(features, assumed, p, S.risk_tier(p, self.threshold), p >= self.threshold, self.gain(p),
                            self._range_warnings(features))
        if assumed:
            result.warnings.append("Assumed for missing details: " + ", ".join(
                f"{S.LABELS[k].lower()} = {self._show(k, v)}" for k, v in assumed.items()) + ".")
        if explain:
            result.factors = self._factors(features, p)
            result.actions = self._actions(features, p)
        if similar:
            result.similar = self._similar(features)
        return result

    @staticmethod
    def _show(name, value):
        if name in S.BINARY:
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:,.1f}".rstrip("0").rstrip(".")
        return str(value)

    # ------------------------------------------------------------------ explanations
    def _factors(self, features, p, top=6):
        """How many points each detail adds to (or takes from) the churn chance compared with a typical customer.

        Each detail is swapped, one at a time, for its typical value (the median, or the most common category)
        and the model re-scores the customer; the difference is that detail's effect.
        """
        variants, names = [], []
        for name in S.RAW_FEATURES:
            typical = self._typical[name]
            if name in S.NUMERIC and S.NUMERIC[name][0] == "int":
                typical = int(round(typical))
            if features[name] != typical:
                variants.append({**features, name: typical})
                names.append((name, typical))
        if not variants:
            return []
        rows = []
        for (name, typical), alt in zip(names, self._proba(variants)):
            effect = p - float(alt)
            rows.append({"feature": name, "label": S.LABELS[name], "value": self._show(name, features[name]),
                         "typical": self._show(name, typical), "effect_points": round(100 * effect, 1)})
        rows = [r for r in rows if abs(r["effect_points"]) >= 100 * MIN_EFFECT]
        return sorted(rows, key=lambda r: abs(r["effect_points"]), reverse=True)[:top]

    def _actions(self, features, p, top=4):
        """Things the company could change, ranked by how far each lowers this customer's churn chance."""
        f = features
        options = []
        if f["contract"] == "Month-to-month":
            options.append(("Move them to a one-year contract", {"contract": "One year"}))
            options.append(("Move them to a two-year contract", {"contract": "Two year"}))
        elif f["contract"] == "One year":
            options.append(("Move them to a two-year contract", {"contract": "Two year"}))
        if f["internet_service"] != "None":
            if not f["tech_support"]:
                options.append(("Add the tech-support add-on", {"tech_support": 1}))
            if not f["online_security"]:
                options.append(("Add the online-security add-on", {"online_security": 1}))
        if f["payment_method"] == "Manual payment":
            options.append(("Switch them to automatic payment", {"payment_method": "Bank transfer (auto)"}))
        if f["monthly_charges"] > 10 / 0.9:
            options.append((f"Give a 10% price reduction ({f['monthly_charges']:,.0f} → {f['monthly_charges'] * 0.9:,.0f} a month)",
                            {"monthly_charges": round(f["monthly_charges"] * 0.9, 2)}))
        if f["support_calls"] >= 3:
            options.append((f"Fix their support issues (calls {f['support_calls']} → 1)", {"support_calls": 1}))
        if f["late_payments"] >= 2:
            options.append((f"Sort out billing problems (late payments {f['late_payments']} → 0)", {"late_payments": 0}))
        if not options:
            return []
        value = self.economics["value"]
        rows = []
        for (text, change), new in zip(options, self._proba([{**f, **change} for _, change in options])):
            drop = p - float(new)
            rows.append({"action": text, "new_probability": round(float(new), 4), "reduction_points": round(100 * drop, 1),
                         "retained_value": round(drop * value, 0)})
        rows = [r for r in rows if r["reduction_points"] >= 100 * MIN_EFFECT * 2]
        return sorted(rows, key=lambda r: r["reduction_points"], reverse=True)[:top]

    def what_if(self, raw, feature, values):
        """Churn chance for the same customer with one detail changed. Returns [(value, probability), ...]."""
        feature = feature.strip().lower().replace(" ", "_")
        if feature not in S.RAW_FEATURES:
            raise ValidationError(f"Unknown feature '{feature}'. Choose one of: {', '.join(S.RAW_FEATURES)}.")
        features, _ = parse_features(raw)
        variants, shown = [], []
        for value in values:
            try:
                changed, _ = parse_features({**features, feature: value})
            except ValidationError as exc:
                raise ValidationError(f"{value!r} does not work for {S.LABELS[feature].lower()}: "
                                      f"{exc.errors.get(feature) or exc}") from None
            variants.append(changed)
            shown.append(changed[feature])
        return list(zip(shown, self._proba(variants).tolist()))

    def comparison_values(self, feature):
        """Sensible values to try in a what-if for a feature."""
        if feature in S.CATEGORICAL:
            return list(S.CATEGORICAL[feature])
        if feature in S.BINARY:
            return [0, 1]
        kind, low, high, _ = S.NUMERIC[feature]
        lo, hi = self._ranges[feature]
        points = np.linspace(lo, hi, 6)
        return sorted({int(round(v)) if kind == "int" else round(float(v), 1) for v in points})

    def _similar(self, features, n=3):
        """The training customers closest to this one (same contract preferred)."""
        ref = self.reference
        cols = ("tenure_months", "monthly_charges", "support_calls")
        scale = ref[list(cols)].std().replace(0, 1)
        dist = np.sqrt(sum(((ref[c].fillna(ref[c].median()) - features[c]) / scale[c]) ** 2 for c in cols))
        dist = dist + (ref["contract"] != features["contract"]) * 1.0 + (ref["internet_service"] != features["internet_service"]) * 0.5
        rows = []
        for idx in dist.nsmallest(n).index:
            r = ref.loc[idx]
            rows.append({"contract": r["contract"], "internet_service": r["internet_service"],
                         "tenure_months": None if pd.isna(r["tenure_months"]) else int(r["tenure_months"]),
                         "monthly_charges": None if pd.isna(r["monthly_charges"]) else round(float(r["monthly_charges"]), 1),
                         "support_calls": None if pd.isna(r["support_calls"]) else int(r["support_calls"]),
                         "outcome": "Churned" if int(r[S.TARGET]) == 1 else "Stayed"})
        return rows

    # ------------------------------------------------------------------ batch
    def predict_csv(self, in_path, out_path):
        """Score every row of a CSV. Bad rows are reported (with line numbers), good rows still get a score.

        Returns (ok_count, errors, summary). The output CSV has the original columns plus churn_probability,
        risk_tier, make_offer, expected_gain, priority_rank (1 = most valuable offer) and a status column.
        """
        in_path, out_path = Path(str(in_path).strip().strip('"')), Path(str(out_path).strip().strip('"'))
        if not in_path.exists():
            raise DataError(f"File not found: {in_path}")
        if out_path.exists():
            raise DataError(f"{out_path} already exists; choose a new file name.")
        try:
            df = pd.read_csv(in_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            raise DataError(f"{in_path.name} is empty.") from None
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            raise DataError(f"Could not read {in_path.name}: {exc}") from None
        df.columns = normalise_headers(df.columns)
        if len(df) > MAX_BATCH_ROWS:
            raise DataError(f"{in_path.name} has {len(df)} rows; the limit is {MAX_BATCH_ROWS}.")
        missing = [c for c in REQUIRED_INPUT if c not in df.columns]
        if missing:
            raise DataError(f"{in_path.name} is missing required column(s): {', '.join(missing)}.")

        parsed, errors, status = [], [], []
        for position, row in enumerate(df.to_dict("records")):
            line = position + 2   # header is line 1
            try:
                features, assumed = parse_features({k: v for k, v in row.items() if k in S.RAW_FEATURES})
                parsed.append((position, features))
                notes = (["defaults used"] if assumed else []) + \
                        (["outside the range the model learned: less reliable"] if self._range_warnings(features) else [])
                status.append("ok" + (f" ({'; '.join(notes)})" if notes else ""))
            except ValidationError as exc:
                errors.append(f"Line {line}: {exc}")
                status.append(f"error: {exc}")
                parsed.append((position, None))
        valid = [(i, f) for i, f in parsed if f is not None]
        out = df.copy()
        for col in ("churn_probability", "risk_tier", "make_offer", "expected_gain", "priority_rank"):
            out[col] = ""
        summary = {"scored": len(valid), "high": 0, "medium": 0, "low": 0, "offers": 0, "expected_gain": 0.0}
        if valid:
            probabilities = self._proba([f for _, f in valid])
            gains = np.array([self.gain(p) for p in probabilities])
            order = {i: rank for rank, i in enumerate((valid[k][0] for k in np.argsort(-gains, kind="stable")), 1)}
            for (i, _), p, g in zip(valid, probabilities, gains):
                tier = S.risk_tier(float(p), self.threshold)
                offer = bool(p >= self.threshold)
                out.loc[i, "churn_probability"] = f"{p:.4f}"
                out.loc[i, "risk_tier"] = tier
                out.loc[i, "make_offer"] = "yes" if offer else "no"
                out.loc[i, "expected_gain"] = f"{g:.0f}"
                out.loc[i, "priority_rank"] = str(order[i])
                summary[tier.lower()] += 1
                if offer:
                    summary["offers"] += 1
                    summary["expected_gain"] += float(g)
        out["status"] = status
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)
        except OSError as exc:
            raise DataError(f"Could not write {out_path}: {exc.strerror or exc}") from exc
        return len(valid), errors, summary


def require_predictor(directory):
    """Load the saved model or raise NotTrainedError with instructions."""
    if not persistence.is_trained(directory):
        raise NotTrainedError(f"No trained model found in '{directory}'. Train one first "
                              "(menu option 'Train a model', or: python main.py --train).")
    return Predictor.from_directory(directory)
