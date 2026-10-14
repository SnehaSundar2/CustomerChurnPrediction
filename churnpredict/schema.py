"""The dataset schema: every feature, its type, valid range and allowed values.

Training-data cleaning and prediction-input validation both read from here, so the two can never
disagree about what a valid customer looks like.

Fairness note: the model deliberately does NOT use gender, age, marital status or any other personal
attribute. It uses only the plan, billing and service history a company already has about the account.
"""

TARGET = "churn"                 # did the customer leave? 1 = churned, 0 = stayed
ID = "customer_id"               # optional column in a dataset; never used as a feature

CONTRACTS = ["Month-to-month", "One year", "Two year"]
INTERNET = ["None", "DSL", "Fiber"]
PAYMENTS = ["Credit card (auto)", "Bank transfer (auto)", "UPI autopay", "Manual payment"]
AUTO_PAYMENTS = PAYMENTS[:3]

# name -> (type, low, high, description). Numeric limits are hard limits for valid data.
NUMERIC = {
    "tenure_months": ("int", 0, 72, "Months as a customer (0 = joined this month)"),
    "monthly_charges": ("float", 10, 150, "Current monthly bill in rupees (10 to 150 in this dataset's units)"),
    "total_charges": ("float", 0, 12000, "Total billed so far, in the same units"),
    "support_calls": ("int", 0, 20, "Calls to customer support in the last 3 months"),
    "late_payments": ("int", 0, 12, "Late payments in the last 12 months"),
}
BINARY = {
    "paperless_billing": "Receives paperless bills (yes/no)",
    "tech_support": "Has the tech-support add-on (yes/no)",
    "online_security": "Has the online-security add-on (yes/no)",
}
CATEGORICAL = {
    "contract": CONTRACTS,
    "internet_service": INTERNET,
    "payment_method": PAYMENTS,
}

NUMERIC_FEATURES = list(NUMERIC)
BINARY_FEATURES = list(BINARY)
CATEGORICAL_FEATURES = list(CATEGORICAL)
RAW_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = RAW_FEATURES + [TARGET]

# Engineered features added by preprocessing.add_features (never use the target).
ENGINEERED = ["avg_monthly_spend", "price_change", "protection_services"]

MIN_ROWS = 200
MIN_PER_CLASS = 30
TOTAL_RATIO = (0.3, 3.0)         # total charges must be within this multiple of tenure x monthly charges

# Campaign economics (rupees). They decide the contact threshold; change them with --customer-value etc.
CUSTOMER_VALUE = 12000.0         # revenue kept over the next year when a customer who would leave is saved
OFFER_COST = 600.0               # cost of contacting a customer with a retention offer
SAVE_RATE = 0.30                 # share of would-be churners who accept the offer and stay

LABELS = {
    "tenure_months": "Tenure", "monthly_charges": "Monthly charges", "total_charges": "Total charges",
    "support_calls": "Support calls (3 months)", "late_payments": "Late payments (12 months)",
    "paperless_billing": "Paperless billing", "tech_support": "Tech support", "online_security": "Online security",
    "contract": "Contract", "internet_service": "Internet service", "payment_method": "Payment method",
}
UNITS = {"tenure_months": "months", "monthly_charges": "₹", "total_charges": "₹"}

# Alternative spellings people actually use (all lower case, '-' and '_' treated as spaces)
ALIASES = {
    "contract": {"month to month": "Month-to-month", "monthly": "Month-to-month", "mtm": "Month-to-month",
                 "m2m": "Month-to-month", "1 year": "One year", "annual": "One year", "yearly": "One year",
                 "1yr": "One year", "2 year": "Two year", "two years": "Two year", "2yr": "Two year",
                 "biennial": "Two year"},
    "internet_service": {"no": "None", "no internet": "None", "no internet service": "None", "nil": "None",
                         "fibre": "Fiber", "fiber optic": "Fiber", "fibre optic": "Fiber", "ftth": "Fiber",
                         "adsl": "DSL"},
    "payment_method": {"credit card": "Credit card (auto)", "card": "Credit card (auto)", "cc": "Credit card (auto)",
                       "bank transfer": "Bank transfer (auto)", "bank": "Bank transfer (auto)",
                       "net banking": "Bank transfer (auto)", "upi": "UPI autopay", "autopay": "UPI autopay",
                       "upi auto": "UPI autopay", "manual": "Manual payment", "cheque": "Manual payment",
                       "check": "Manual payment", "electronic check": "Manual payment", "cash": "Manual payment"},
}

CHURN_YES = {"yes", "y", "1", "true", "churned", "left", "cancelled", "canceled", "lost"}
CHURN_NO = {"no", "n", "0", "false", "retained", "stayed", "active", "kept"}


def describe(feature):
    if feature in NUMERIC:
        return NUMERIC[feature][3]
    if feature in BINARY:
        return BINARY[feature]
    return {"contract": "Contract type (Month-to-month, One year, Two year)",
            "internet_service": "Internet service (None, DSL, Fiber)",
            "payment_method": "How the bill is paid"}.get(feature, feature)


def risk_tier(probability, contact_threshold):
    """'High' at or above the contact threshold, 'Medium' from half of it, otherwise 'Low'."""
    if probability >= contact_threshold:
        return "High"
    if probability >= contact_threshold / 2:
        return "Medium"
    return "Low"


def _key(text):
    return " ".join(str(text).replace("_", " ").replace("-", " ").split()).lower()


def canonical_category(feature, value):
    """'  month to month ' -> 'Month-to-month'. Returns None when the value is missing or not recognised."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    key = _key(value)
    lookup = {_key(name): name for name in CATEGORICAL[feature]}
    lookup.update({_key(alias): name for alias, name in ALIASES.get(feature, {}).items()})
    return lookup.get(key)


def parse_yes_no(value):
    """'Yes' / 'y' / true / 1 -> 1;  'no' / 'n' / false / 0 -> 0;  anything else -> None."""
    if isinstance(value, (int, float)) and value == value:
        return {1: 1, 0: 0}.get(int(value)) if value in (0, 1) else None
    return {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}.get(str(value).strip().lower())


def canonical_label(value):
    """'Yes ' / 1 / 'churned' -> 1;  'No' / 0 / 'retained' -> 0;  anything else (or missing) -> None."""
    if value is None or (isinstance(value, float) and value != value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and value == int(value):
        value = int(value)
    key = _key(value)
    if key in CHURN_YES:
        return 1
    if key in CHURN_NO:
        return 0
    return None


def format_money(amount):
    """1234 -> '₹1,234'; 315000 -> '₹3.15 lakh'; 12500000 -> '₹1.25 crore'; negatives keep their sign."""
    sign, amount = ("-" if amount < 0 else ""), abs(amount)
    if amount >= 10_000_000:
        return f"{sign}₹{amount / 10_000_000:.2f} crore"
    if amount >= 100_000:
        return f"{sign}₹{amount / 100_000:.2f} lakh"
    return f"{sign}₹{amount:,.0f}"
