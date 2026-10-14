"""Data preprocessing: loading, cleaning, feature engineering and the scikit-learn transformer.

Two kinds of preprocessing, kept deliberately separate:

* clean()               — fixes or removes BAD DATA (typos, impossible values, impossible combinations, repeated
                          customers) before the split. Every rule looks at one row (or at exact repeats); none learns
                          statistics from other rows, so nothing from the test set can influence training.
* build_preprocessor()  — LEARNS from data (medians for imputing, means for scaling, categories for one-hot
                          encoding). It lives inside the model pipeline and is fitted on the TRAINING rows only.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import schema as S
from .exceptions import DataError

log = logging.getLogger("churnpredict")

MAX_FILE_BYTES = 50_000_000
EXAMPLE_SPELLING = {"contract": "e.g. 'MTM' → 'Month-to-month'", "internet_service": "e.g. 'fibre' → 'Fiber'",
                    "payment_method": "e.g. 'UPI' → 'UPI autopay'"}


# ====================================================================== loading
def normalise_headers(columns):
    return [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in columns]


def load_dataset(path):
    """Read a CSV and check it has the columns the project needs (customer_id is optional)."""
    path = Path(str(path).strip().strip('"'))
    if not path.exists() or not path.is_file():
        raise DataError(f"Dataset file not found: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DataError(f"{path.name} is larger than {MAX_FILE_BYTES // 1_000_000} MB.")
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=object, keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError:
        raise DataError(f"{path.name} is empty.") from None
    except UnicodeDecodeError:
        raise DataError(f"{path.name} is not UTF-8 text. In Excel use 'Save As > CSV UTF-8'.") from None
    except (pd.errors.ParserError, OSError) as exc:
        raise DataError(f"Could not read {path.name}: {exc}") from None
    df.columns = normalise_headers(df.columns)
    missing = [c for c in S.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"{path.name} is missing column(s): {', '.join(missing)}. "
                        f"Required columns: {', '.join(S.REQUIRED_COLUMNS)} (customer_id is optional).")
    if len(df) < S.MIN_ROWS:
        raise DataError(f"{path.name} has only {len(df)} rows; at least {S.MIN_ROWS} are needed to train a "
                        "reliable model.")
    columns = ([S.ID] if S.ID in df.columns else []) + S.REQUIRED_COLUMNS
    return df[columns].copy()


# ====================================================================== cleaning
@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    steps: list = field(default_factory=list)        # (label, count, outcome)
    missing_before: dict = field(default_factory=dict)
    missing_after: dict = field(default_factory=dict)
    class_counts: dict = field(default_factory=dict)

    def add(self, label, count, outcome):
        if count:
            self.steps.append((label, int(count), outcome))

    @property
    def rows_dropped(self):
        return self.rows_in - self.rows_out

    def lines(self):
        out = [f"{self.rows_in} rows read → {self.rows_out} kept ({self.rows_dropped} removed)"]
        out += [f"  • {label}: {count} — {outcome}" for label, count, outcome in self.steps]
        if self.class_counts:
            total = sum(self.class_counts.values())
            out.append(f"  Kept: {self.class_counts['churned']} churned and {self.class_counts['stayed']} stayed "
                       f"({100 * self.class_counts['churned'] / total:.1f}% churn)")
        return out

    def to_dict(self):
        return {"rows_in": self.rows_in, "rows_out": self.rows_out,
                "steps": [{"label": a, "count": b, "outcome": c} for a, b, c in self.steps],
                "missing_before": self.missing_before, "missing_after": self.missing_after,
                "class_counts": self.class_counts}


def clean(df):
    """Return (clean_df with int `churn`, CleaningReport). The input is not modified."""
    has_id = S.ID in df.columns
    df = df.copy()
    report = CleaningReport(rows_in=len(df))
    report.missing_before = {c: int(n) for c, n in df.isna().sum().items() if n}

    # 1. the label: a customer whose outcome is unknown cannot teach the model anything
    label = df[S.TARGET].map(S.canonical_label)
    report.add("Rows with a missing or unrecognised churn label", label.isna().sum(), "removed")
    df = df[label.notna()].copy()
    df[S.TARGET] = label[label.notna()].astype(int)

    # 2. categories: fix spelling; a missing contract is not guessable, other gaps are filled in later
    for col in S.CATEGORICAL_FEATURES:
        original = df[col]
        canon = original.map(lambda v, c=col: S.canonical_category(c, v))
        report.add(f"{S.LABELS[col]} spelling standardised",
                   (canon.notna() & (canon != original.map(lambda v: v if isinstance(v, str) else None))).sum(),
                   EXAMPLE_SPELLING[col])
        unrecognised = canon.isna() & original.notna()
        df[col] = canon
        if col == "contract":
            report.add("Rows with an unknown or missing contract", canon.isna().sum(), "removed")
            df = df[canon.notna()]
        else:
            report.add(f"Unrecognised {S.LABELS[col].lower()} values", unrecognised.sum(), "set to missing, filled in later")

    # 3. numbers: text → numbers, yes/no → 1/0
    blank_total = df["total_charges"].map(lambda v: isinstance(v, str) and not v.strip())
    for col in S.NUMERIC_FEATURES:
        before = df[col].isna() | (df[col].map(lambda v: isinstance(v, str) and not v.strip()))
        df[col] = pd.to_numeric(df[col], errors="coerce")
        report.add(f"Non-numeric text in {col}", (df[col].isna() & ~before).sum(), "set to missing")
    for col in S.BINARY_FEATURES:
        flags = df[col].map(S.parse_yes_no)
        report.add(f"Unrecognised {S.LABELS[col].lower()} values", (flags.isna() & df[col].notna()).sum(), "set to missing")
        df[col] = flags
    df = df.replace([np.inf, -np.inf], np.nan)
    new_customer = blank_total & (df["tenure_months"] == 0)
    report.add("Blank total charges for brand-new customers (tenure 0)", new_customer.sum(), "set to 0")
    df.loc[new_customer, "total_charges"] = 0.0

    # 4. out-of-range values are probably typos: blank them and impute later
    for col in S.NUMERIC_FEATURES:
        lo, hi = S.NUMERIC[col][1:3]
        bad = df[col].notna() & ~df[col].between(lo, hi)
        report.add(f"Out-of-range {col}", bad.sum(), f"set to missing (valid: {lo}–{hi})")
        df.loc[bad, col] = np.nan

    # 5. combinations that cannot be true
    expected = df["tenure_months"] * df["monthly_charges"]
    lo_r, hi_r = S.TOTAL_RATIO
    mismatch = (df["tenure_months"] >= 3) & df["total_charges"].notna() & expected.notna() & \
        ~df["total_charges"].between(expected * lo_r, expected * hi_r)
    report.add("Total charges that cannot match tenure × monthly charges", mismatch.sum(), "total set to missing")
    df.loc[mismatch, "total_charges"] = np.nan
    addons = (df["internet_service"] == "None") & ((df["tech_support"] == 1) | (df["online_security"] == 1))
    report.add("Internet add-ons on an account with no internet service", addons.sum(), "add-ons set to no")
    df.loc[addons, ["tech_support", "online_security"]] = 0

    # 6. repeats
    dup = df.duplicated()
    report.add("Duplicate rows", dup.sum(), "removed (kept the first copy)")
    df = df[~dup]
    if has_id:
        same_id = df[S.ID].notna() & df[S.ID].duplicated()
        report.add("Customer id appearing more than once with different details", same_id.sum(), "removed (kept the first)")
        df = df[~same_id]

    df = df.reset_index(drop=True)
    counts = df[S.TARGET].value_counts()
    report.class_counts = {"churned": int(counts.get(1, 0)), "stayed": int(counts.get(0, 0))}
    report.rows_out = len(df)
    report.missing_after = {c: int(n) for c, n in df.isna().sum().items() if n and c != S.ID}
    log.info("Cleaning: %d → %d rows", report.rows_in, report.rows_out)
    if len(df) < S.MIN_ROWS:
        raise DataError(f"Only {len(df)} usable rows remain after cleaning (need {S.MIN_ROWS}). "
                        "Check that the column names and units match the schema.")
    for name, count in report.class_counts.items():
        if count < S.MIN_PER_CLASS:
            raise DataError(f"Only {count} customers who {name} remain after cleaning; at least {S.MIN_PER_CLASS} of each "
                            "kind are needed to train and test a model.")
    return df, report


# ====================================================================== features
def add_features(df):
    """Engineered features. They use only other input columns — never the churn label (that would be leakage)."""
    out = pd.DataFrame(df).copy()
    months = out["tenure_months"].clip(lower=1)
    out["avg_monthly_spend"] = out["total_charges"] / months                     # what they have really paid per month
    out["price_change"] = out["monthly_charges"] - out["avg_monthly_spend"]      # positive = the bill went up
    out["protection_services"] = out["tech_support"] + out["online_security"]
    return out


def build_preprocessor():
    """Impute → scale numbers, impute binary flags, impute → one-hot encode categories."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([
        ("num", numeric, S.NUMERIC_FEATURES + S.ENGINEERED),
        ("bin", binary, S.BINARY_FEATURES),
        ("cat", categorical, S.CATEGORICAL_FEATURES),
    ])


def feature_adder():
    return FunctionTransformer(add_features)


def split_xy(df):
    return df[S.RAW_FEATURES].copy(), df[S.TARGET].astype(int).copy()
