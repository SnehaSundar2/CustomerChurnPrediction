"""Validation of customer records (prediction input) and of campaign settings.

parse_features() turns raw text values (from the menu, a web form, a JSON request or a CSV row) into a clean,
consistent customer record — or raises ValidationError listing EVERY problem at once, with a message per field.
It uses the same schema as training-data cleaning, so both agree on what is valid.
"""

import difflib
import math
import re

from . import schema as S
from .exceptions import ValidationError

REQUIRED_INPUT = ["tenure_months", "contract", "monthly_charges", "internet_service"]
DEFAULTS = {"support_calls": 1, "late_payments": 0, "paperless_billing": 0, "tech_support": 0, "online_security": 0,
            "payment_method": "Bank transfer (auto)"}     # total_charges defaults to tenure × monthly charges

_NUMBER = re.compile(r"^(?:rs\.?|₹)?\s*([-+]?\d+(?:\.\d+)?|[-+]?\.\d+)\s*(months?|mos?|calls?|₹)?$", re.I)


def normalise_key(key):
    return str(key).strip().lower().replace(" ", "_").replace("-", "_")


def parse_number(feature, value):
    """'₹1,299.50' → 1299.5 ; '14 months' → 14 (whole numbers for count features). Raises ValueError with a message."""
    kind, low, high, _ = S.NUMERIC[feature]
    label = S.LABELS[feature]
    unit = f" {S.UNITS[feature]}" if feature in S.UNITS and S.UNITS[feature] != "₹" else ""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a number.")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = _NUMBER.match(str(value).strip().replace(",", ""))
        if not match:
            raise ValueError(f"{label} must be a number (you entered '{str(value).strip()[:30]}').")
        number = float(match.group(1))
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a normal number.")
    if kind == "int":
        if number != int(number):
            raise ValueError(f"{label} must be a whole number (you entered {number:g}).")
        number = int(number)
    if not low <= number <= high:
        raise ValueError(f"{label} must be between {low:g}{unit} and {high:g}{unit} (you entered {number:g}).")
    return number


def parse_category(feature, value):
    canon = S.canonical_category(feature, value)
    if canon is None:
        allowed = S.CATEGORICAL[feature]
        guess = difflib.get_close_matches(str(value).strip().title(), allowed, n=1, cutoff=0.6)
        hint = f" Did you mean '{guess[0]}'?" if guess else ""
        raise ValueError(f"{S.LABELS[feature]} '{str(value).strip()[:30]}' is not supported.{hint} "
                         f"Choose one of: {', '.join(allowed)}.")
    return canon


def parse_features(raw, apply_defaults=True):
    """Validate a customer record. Returns (features, assumed) where `assumed` lists any defaults used.

    Raises ValidationError(message, errors={field: message}) listing all problems.
    """
    if not isinstance(raw, dict):
        raise ValidationError("The customer record must be a set of named values.")
    given = {normalise_key(k): v for k, v in raw.items()}
    errors = {}
    unknown = sorted(set(given) - set(S.RAW_FEATURES) - {S.ID})
    if unknown:
        errors["_"] = f"Unknown field(s): {', '.join(unknown)}. Valid fields: {', '.join(S.RAW_FEATURES)}."
    features, assumed = {}, {}

    for name in S.RAW_FEATURES:
        value = given.get(name)
        blank = value is None or (isinstance(value, str) and not value.strip()) or \
            (isinstance(value, float) and value != value)
        if blank:
            if name in REQUIRED_INPUT or (not apply_defaults and name != "total_charges"):
                errors[name] = f"{S.LABELS[name]} is required."
            elif name in DEFAULTS and apply_defaults:
                features[name] = DEFAULTS[name]
                assumed[name] = DEFAULTS[name]
            continue
        try:
            if name in S.NUMERIC:
                features[name] = parse_number(name, value)
            elif name in S.CATEGORICAL:
                features[name] = parse_category(name, value)
            else:
                flag = S.parse_yes_no(value)
                if flag is None:
                    raise ValueError(f"{S.LABELS[name]} must be yes or no.")
                features[name] = flag
        except ValueError as exc:
            errors[name] = str(exc)

    f = features
    if "total_charges" not in f and not errors and apply_defaults:
        f["total_charges"] = round(f["tenure_months"] * f["monthly_charges"], 2)
        assumed["total_charges"] = f["total_charges"]
    if not errors:   # cross-field checks only make sense when every value parsed
        if f["internet_service"] == "None":
            for name in ("tech_support", "online_security"):
                if f[name]:
                    errors[name] = f"{S.LABELS[name]} needs an internet service, but the customer has none."
        expected = f["tenure_months"] * f["monthly_charges"]
        low, high = S.TOTAL_RATIO
        if f["tenure_months"] >= 3 and not expected * low <= f["total_charges"] <= expected * high:
            errors["total_charges"] = (f"Total charges ({f['total_charges']:,.0f}) do not fit {f['tenure_months']} months at "
                                       f"{f['monthly_charges']:,.0f} a month (expected roughly {expected * low:,.0f} to {expected * high:,.0f}).")
        if f["tenure_months"] < 3 and f["total_charges"] > f["monthly_charges"] * 4:
            errors["total_charges"] = (f"Total charges ({f['total_charges']:,.0f}) are too high for a customer of "
                                       f"{f['tenure_months']} month(s).")
    if errors:
        raise ValidationError(" ".join(message.rstrip(".") + "." for message in errors.values()), errors)
    return features, assumed


def validate_economics(value, cost, save_rate):
    """Campaign settings must be positive; the save rate is a share between 0 and 1."""
    errors = {}
    for name, number in (("customer value", value), ("offer cost", cost)):
        if not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0:
            errors[name] = f"The {name} must be a positive number."
    if not isinstance(save_rate, (int, float)) or not 0 < save_rate <= 1:
        errors["save rate"] = "The save rate must be above 0 and at most 1 (for example 0.3 for 30%)."
    if errors:
        raise ValidationError(" ".join(errors.values()), errors)
    return float(value), float(cost), float(save_rate)


def validate_port(value):
    try:
        port = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError("Port must be a whole number.") from None
    if not 1024 <= port <= 65535:
        raise ValidationError("Port must be between 1024 and 65535.")
    return port


def validate_int(value, field, low, high):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a whole number.") from None
    if not low <= number <= high:
        raise ValidationError(f"{field} must be between {low} and {high}.")
    return number


def validate_number(value, field, low, high):
    try:
        number = float(str(value).strip().replace(",", ""))
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number.") from None
    if not math.isfinite(number) or not low <= number <= high:
        raise ValidationError(f"{field} must be between {low:g} and {high:g}.")
    return number
