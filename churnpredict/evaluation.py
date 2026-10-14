"""Model evaluation for churn: ranking quality, honest probabilities, and what a retention campaign earns.

A churn model is only worth its money if acting on it pays. So besides the usual metrics this module
chooses the contact threshold from the campaign economics (cost of an offer, value of a saved customer,
share of offers that work), compares the model with "contact nobody" and "contact everybody", and builds the
lift and calibration tables.
"""

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from . import schema as S


def classification_metrics(actual, probability, threshold):
    """Counts and percentages when every customer scoring at or above `threshold` is flagged."""
    actual = np.asarray(actual, int)
    flagged = np.asarray(probability, float) >= threshold
    churned = actual == 1
    tp, fp = int((flagged & churned).sum()), int((flagged & ~churned).sum())
    fn, tn = int((~flagged & churned).sum()), int((~flagged & ~churned).sum())
    precision = 100 * tp / (tp + fp) if tp + fp else 0.0
    recall = 100 * tp / (tp + fn) if tp + fn else 0.0
    return {
        "n": int(len(actual)), "threshold": float(threshold), "flagged": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": 100 * (tp + tn) / len(actual),
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def ranking_metrics(actual, probability):
    """Threshold-free scores: ROC AUC, average precision (PR AUC) and the Brier score (lower = more honest)."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    if len(set(actual.tolist())) < 2:
        return {"roc_auc": 0.0, "average_precision": 0.0, "brier": float(brier_score_loss(actual, probability))}
    return {"roc_auc": float(roc_auc_score(actual, probability)),
            "average_precision": float(average_precision_score(actual, probability)),
            "brier": float(brier_score_loss(actual, probability))}


# ---------------------------------------------------------------------- campaign economics
def campaign_profit(actual, probability, threshold, value=S.CUSTOMER_VALUE, cost=S.OFFER_COST, save_rate=S.SAVE_RATE):
    """Expected profit of offering a retention deal to everyone at or above `threshold`.

    Every contacted customer costs `cost`. A contacted customer who really would leave is kept with chance
    `save_rate`, which is worth `value`. Customers who would have stayed anyway still cost the offer.
    """
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    contacted = probability >= threshold
    churners = int((contacted & (actual == 1)).sum())
    profit = churners * save_rate * value - int(contacted.sum()) * cost
    return {"contacted": int(contacted.sum()), "churners_reached": churners, "saved": churners * save_rate,
            "profit": float(profit), "profit_per_1000": float(profit / len(actual) * 1000)}


def break_even_probability(value=S.CUSTOMER_VALUE, cost=S.OFFER_COST, save_rate=S.SAVE_RATE):
    """The churn chance above which an offer pays for itself on average: cost ÷ (save rate × value)."""
    return cost / (save_rate * value)


def choose_contact_threshold(actual, probability, value=S.CUSTOMER_VALUE, cost=S.OFFER_COST, save_rate=S.SAVE_RATE):
    """The churn chance above which an offer is made: the break-even point, cost ÷ (save rate × value).

    With honest (calibrated) probabilities this is exactly the profit-maximising rule: contact a customer when the
    expected saving is more than the offer costs. The out-of-fold training predictions are then used to CHECK that
    the campaign really makes money, and the empirically best threshold is reported for comparison (it is usually
    close, but it chases noise, so it is not used).

    Returns {'contact': t, 'profitable': bool, ...}. If the break-even chance is 100% or more, or the campaign
    loses money on the training predictions, t is just above 1 (contact nobody).
    """
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    break_even = break_even_probability(value, cost, save_rate)
    at_break_even = campaign_profit(actual, probability, break_even, value, cost, save_rate)["profit"] if break_even < 1 else 0.0
    best_t, best = 1.0000001, 0.0
    for t in np.unique(probability):
        profit = campaign_profit(actual, probability, t, value, cost, save_rate)["profit"]
        if profit > best + 1e-9:
            best_t, best = float(t), profit
    profitable = break_even < 1 and at_break_even > 0
    return {"contact": break_even if profitable else 1.0000001, "profitable": bool(profitable),
            "training_profit_per_1000": max(at_break_even, 0.0) / len(actual) * 1000 if profitable else 0.0,
            "break_even": break_even, "empirical_best": best_t,
            "empirical_best_profit_per_1000": best / len(actual) * 1000,
            "value": float(value), "cost": float(cost), "save_rate": float(save_rate)}


def profit_curve(actual, probability, value, cost, save_rate, steps=50):
    """[(threshold, profit per 1000 customers)] for a chart of how the choice of threshold matters."""
    return [(round(t, 3), round(campaign_profit(actual, probability, t, value, cost, save_rate)["profit_per_1000"], 1))
            for t in np.linspace(0, 1, steps + 1)]


def policy_table(actual, probability, threshold, value, cost, save_rate):
    """Contact nobody / everybody / the model's list / a perfect crystal ball, side by side."""
    actual = np.asarray(actual, int)
    ones = np.ones(len(actual))
    perfect = campaign_profit(actual, actual.astype(float), 1, value, cost, save_rate)
    rows = [("Contact nobody", {"contacted": 0, "churners_reached": 0, "saved": 0.0, "profit": 0.0, "profit_per_1000": 0.0}),
            ("Contact everybody", campaign_profit(actual, ones, 0.5, value, cost, save_rate)),
            ("Use the model", campaign_profit(actual, probability, threshold, value, cost, save_rate)),
            ("Perfect foresight (impossible)", perfect)]
    return [{"policy": name, **vals} for name, vals in rows]


# ---------------------------------------------------------------------- tables for the report
def lift_table(actual, probability, bins=10):
    """Customers sorted from most to least likely to churn and cut into equal groups (deciles)."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    order = np.argsort(-probability, kind="stable")
    groups = np.array_split(order, bins)
    overall = actual.mean() or 1.0
    total = max(actual.sum(), 1)
    rows, captured = [], 0
    for i, idx in enumerate(groups, 1):
        churned = int(actual[idx].sum())
        captured += churned
        rate = churned / len(idx) if len(idx) else 0.0
        rows.append({"decile": i, "customers": int(len(idx)), "churned": churned, "churn_rate": 100 * rate,
                     "lift": rate / overall, "captured": 100 * captured / total,
                     "mean_probability": 100 * float(probability[idx].mean()) if len(idx) else 0.0})
    return rows


def gains_points(actual, probability, steps=20):
    """[(share of customers contacted, share of churners reached)] when contacting in order of risk."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    order = np.argsort(-probability, kind="stable")
    caught = np.cumsum(actual[order]) / max(actual.sum(), 1)
    cuts = np.unique(np.linspace(0, len(actual), steps + 1).round().astype(int))
    return [(round(c / len(actual), 4), round(float(caught[c - 1]) if c else 0.0, 4)) for c in cuts]


def calibration_table(actual, probability, bins=10):
    """Predicted chance against what really happened, per probability band. Empty bands are skipped."""
    actual, probability = np.asarray(actual, int), np.asarray(probability, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = which == b
        if mask.any():
            rows.append({"band": f"{edges[b] * 100:.0f}-{edges[b + 1] * 100:.0f}%", "customers": int(mask.sum()),
                         "predicted": 100 * float(probability[mask].mean()), "observed": 100 * float(actual[mask].mean())})
    return rows


def segment_rows(frame, by):
    """Per group: customers, real churn rate, mean predicted churn rate and the gap between them."""
    rows = []
    for key, group in frame.groupby(by, observed=True):
        actual, predicted = 100 * group["actual"].mean(), 100 * group["probability"].mean()
        rows.append({"group": str(key), "n": int(len(group)), "actual": float(actual), "predicted": float(predicted),
                     "gap": float(predicted - actual)})
    return sorted(rows, key=lambda r: -r["actual"])


def feature_importance(pipeline, X_test, y_test, seed=42, repeats=8):
    """Permutation importance: how much ROC AUC drops when one feature's values are shuffled (test set)."""
    result = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc", n_repeats=repeats, random_state=seed,
                                    n_jobs=1)
    rows = [{"feature": f, "label": S.LABELS[f], "importance": max(0.0, float(m)), "std": float(s)}
            for f, m, s in zip(X_test.columns, result.importances_mean, result.importances_std)]
    return sorted(rows, key=lambda r: r["importance"], reverse=True)


def roc_curve_ceiling(df_clean):
    """For SYNTHETIC data only: the ROC AUC of the TRUE churn probabilities against what actually happened.

    No model can systematically beat this: it is the effect of pure luck in who really leaves.
    """
    from .datagen import churn_probability
    d = df_clean
    p = churn_probability(d["tenure_months"].fillna(d["tenure_months"].median()),
                          d["monthly_charges"].fillna(d["monthly_charges"].median()),
                          d["support_calls"].fillna(1), d["late_payments"].fillna(0), d["contract"],
                          d["internet_service"].fillna("Fiber"), d["payment_method"].fillna("Bank transfer (auto)"),
                          d["paperless_billing"].fillna(0), d["tech_support"].fillna(0), d["online_security"].fillna(0))
    return ranking_metrics(d[S.TARGET].map(S.canonical_label), p)
