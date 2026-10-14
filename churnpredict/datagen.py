"""Synthetic customer data for the project.

Real customer records are private, so the project ships a generator that builds a subscription-business
dataset (think mobile or broadband) with a KNOWN churn rule. Each customer's true chance of leaving is
computed from their plan, billing and service history; whether they actually leave is then a random draw
from that chance. Two things follow:

* no model can ever be perfect: even the true rule only ranks customers so well (the "ceiling"), and
* the evaluation can show how close a model gets to that ceiling.

The data is clearly SYNTHETIC: it teaches the workflow and must never be used to judge real customers.
The generated file is also deliberately DIRTY (blank totals, impossible values, messy spellings, repeated
customers) so the preprocessing stage has real work to do.
"""

import numpy as np
import pandas as pd

from . import schema as S

INTERCEPT = -1.8             # tuned so that about a quarter of customers churn
CONTRACT_EFFECT = {"Month-to-month": 0.9, "One year": -0.3, "Two year": -1.3}
INTERNET_EFFECT = {"None": -0.8, "DSL": -0.1, "Fiber": 0.7}
PAYMENT_EFFECT = {"Credit card (auto)": -0.25, "Bank transfer (auto)": -0.25, "UPI autopay": -0.15,
                  "Manual payment": 0.55}


def churn_probability(tenure, monthly, support_calls, late, contract, internet, payment, paperless, tech, security):
    """The 'ground truth' rule: the chance (0-1) that each customer churns. Vectorised over arrays / columns."""
    tenure, monthly = np.asarray(tenure, float), np.asarray(monthly, float)
    calls, late = np.asarray(support_calls, float), np.asarray(late, float)
    paperless, tech, security = (np.asarray(x, float) for x in (paperless, tech, security))
    internet = np.asarray(internet)
    z = (INTERCEPT
         + np.array([CONTRACT_EFFECT[c] for c in contract])
         + 0.6 - 1.6 * (1 - np.exp(-tenure / 20.0))                 # new customers leave more; loyalty builds slowly
         + np.array([INTERNET_EFFECT[i] for i in internet])
         + 0.016 * (monthly - 65)                                    # expensive plans
         + 0.14 * np.minimum(calls, 10) + 0.35 * (calls >= 4)        # unhappy customers call support, a lot
         + 0.22 * late
         + np.array([PAYMENT_EFFECT[p] for p in payment])
         + 0.25 * paperless
         - 0.5 * tech - 0.45 * security
         + 0.4 * ((internet == "Fiber") & (tech == 0)))              # fiber customers without support are the angriest
    return 1 / (1 + np.exp(-z))


def generate(rows=3000, seed=2026, dirty=True):
    """Return a DataFrame with customer_id, the schema's feature columns and churn ('Yes'/'No')."""
    rng = np.random.default_rng(seed)
    contract = rng.choice(S.CONTRACTS, rows, p=[0.55, 0.21, 0.24])
    mean_tenure = np.select([contract == "Month-to-month", contract == "One year"], [14, 30], default=44)
    tenure = np.clip(rng.gamma(2.0, mean_tenure / 2.0), 0, 72).astype(int)
    internet = rng.choice(S.INTERNET, rows, p=[0.2, 0.35, 0.45])
    has_net = internet != "None"
    tech = ((rng.random(rows) < 0.30) & has_net).astype(int)
    security = ((rng.random(rows) < 0.30) & has_net).astype(int)
    base = np.select([internet == "None", internet == "DSL"], [20, 45], default=80)
    monthly = np.clip(base + rng.normal(0, 7, rows) + 5 * tech + 6 * security, 10, 150).round(2)
    total = np.where(tenure == 0, 0.0, tenure * monthly * rng.uniform(0.9, 1.05, rows)).round(2)
    payment = rng.choice(S.PAYMENTS, rows, p=[0.25, 0.20, 0.20, 0.35])
    manual = payment == "Manual payment"
    calls = np.clip(rng.poisson(0.8 + 0.8 * (internet == "Fiber")), 0, 20)
    late = np.clip(rng.poisson(0.4 + 0.7 * manual), 0, 12)
    paperless = (rng.random(rows) < 0.60).astype(int)
    p = churn_probability(tenure, monthly, calls, late, contract, internet, payment, paperless, tech, security)
    churned = rng.random(rows) < p
    df = pd.DataFrame({
        S.ID: [f"CU{10000 + i}" for i in range(rows)], "tenure_months": tenure, "monthly_charges": monthly,
        "total_charges": total, "support_calls": calls, "late_payments": late, "paperless_billing": paperless,
        "tech_support": tech, "online_security": security, "contract": contract, "internet_service": internet,
        "payment_method": payment, S.TARGET: np.where(churned, "Yes", "No")})
    return make_dirty(df, rng) if dirty else df


def make_dirty(df, rng):
    """Inject realistic data-quality problems (about 10% of rows are affected)."""
    df = df.copy().astype(object)
    n = len(df)

    def rows(count):
        return rng.choice(n, count, replace=False)

    for col, rate in (("tenure_months", 0.01), ("monthly_charges", 0.02), ("support_calls", 0.03),
                      ("late_payments", 0.03), ("internet_service", 0.02), ("payment_method", 0.03),
                      ("paperless_billing", 0.02), ("tech_support", 0.02), ("online_security", 0.02)):
        df.loc[rng.random(n) < rate, col] = np.nan                                       # missing values
    new = df["tenure_months"].astype(float) == 0
    df.loc[new, "total_charges"] = " "                                                    # the classic blank total for new customers
    df.loc[rng.random(n) < 0.015, "total_charges"] = ""                                   # other blank totals
    for idx in rows(4):
        df.loc[idx, "tenure_months"] = int(rng.choice([-3, 500, 120]))                    # impossible tenure
    for idx in rows(4):
        df.loc[idx, "monthly_charges"] = float(rng.choice([-20, 0, 900]))                 # impossible bill
    for idx in rows(4):
        df.loc[idx, "support_calls"] = int(rng.choice([999, -1]))
    for idx in rows(3):
        df.loc[idx, "late_payments"] = -2
    for idx in rows(5):
        df.loc[idx, "total_charges"] = 50000                                              # total that cannot match tenure x bill
    for idx in rows(6):                                                                   # add-ons without any internet service
        df.loc[idx, ["internet_service", "tech_support", "online_security"]] = ["None", 1, 1]
    spellings = {"contract": ["month-to-month", "Month to month", "MTM", "1 year", "two year", "Annual"],
                 "internet_service": ["fibre", "Fiber optic", "no internet", "dsl"],
                 "payment_method": ["Credit Card", "UPI", "manual", "bank transfer"]}
    for col, options in spellings.items():
        for idx in np.flatnonzero(rng.random(n) < 0.04):
            df.loc[idx, col] = str(rng.choice(options))
    for idx in rows(5):
        df.loc[idx, "contract"] = np.nan                                                  # unusable: contract is essential
    for col in ("paperless_billing", "tech_support", "online_security"):                  # yes/no written in many ways
        for idx in np.flatnonzero((rng.random(n) < 0.20) & df[col].notna().to_numpy()):
            value = df.loc[idx, col]
            if value in (0, 1):
                df.loc[idx, col] = str(rng.choice(["Yes", "Y", "true", "1"] if value == 1 else ["No", "N", "false", "0"]))
    for idx in np.flatnonzero(rng.random(n) < 0.15):                                      # churn label spellings
        df.loc[idx, S.TARGET] = str(rng.choice(["yes ", "1", "True", "churned"] if df.loc[idx, S.TARGET] == "Yes"
                                               else ["no ", "0", "retained", "False"]))
    for idx in rows(5):
        df.loc[idx, S.TARGET] = np.nan
    for idx in rows(4):
        df.loc[idx, S.TARGET] = "unsure"
    dupes = df.sample(int(n * 0.015), random_state=1)                                     # exact repeats
    reused = df.sample(6, random_state=2).copy()                                          # same customer id, different details
    reused["tenure_months"] = reused["tenure_months"].map(lambda t: t if pd.isna(t) else min(int(t) + 3, 72) if t not in (500, 120) else 10)
    return pd.concat([df, dupes, reused], ignore_index=True).sample(frac=1, random_state=3).reset_index(drop=True)
