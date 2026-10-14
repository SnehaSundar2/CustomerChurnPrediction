# Customer Churn Prediction

A complete machine-learning project that **predicts which customers are about to leave** a subscription
business (mobile, broadband, streaming), says **whether a retention offer is worth making**, and shows
**what could keep each customer**.

It covers the whole workflow the internship brief asks for:

| Brief | Where it lives |
|---|---|
| **Data preprocessing** | `churnpredict/preprocessing.py` – loading, cleaning with an audit log (typos, impossible values, impossible combinations, repeated customers), feature engineering, imputing / scaling / encoding |
| **Model training** | `churnpredict/models.py`, `training.py` – four models plus a baseline compared by stratified cross-validation, tuning, probability calibration, a leak-free split |
| **Evaluation** | `churnpredict/evaluation.py`, `report.py` – ROC AUC, PR AUC, Brier score, lift and gains, calibration, campaign profit against simple policies, fairness-style segment checks, HTML report |
| **Prediction interface** | `cli.py` (menu), `web.py` (web form + JSON API), `main.py` (command line), batch CSV with a ranked call list |

Developed and submitted by **Sneha S** (Intern ID SKX-2026-6425) · Skyrovix virtual internship ·
AI & Machine Learning · due 07 Nov 2026.

> **Data note.** Real customer records are private, so the project generates a realistic but **artificial**
> subscription dataset. Each customer's true chance of leaving comes from a known rule; whether they actually
> leave is a random draw from that chance. That means no model can be perfect, and the report shows how close the
> model gets to the best possible score. Results show the *workflow* works, not accuracy on a real business.
> The model never uses gender, age or any other personal attribute.

## Features

- **Cleaning with an audit trail.** Standardises spellings (`MTM`, `fibre`, `UPI`), yes/no and churn labels; fixes the
  classic blank "total charges" of brand-new customers; blanks impossible values; clears add-ons on accounts with no
  internet; removes duplicate rows and repeated customer ids; says exactly what it did.
- **Fair model comparison.** Baseline ("everyone stays"), Logistic Regression, Decision Tree, Random Forest and Gradient
  Boosting in the same pipeline, scored by 5-fold stratified cross-validation on training rows only.
- **No data leakage.** The test set is used once; imputing, scaling and encoding are fitted inside the pipeline on
  training rows only; engineered features never use the churn label.
- **Real probabilities.** The chosen model is wrapped in a calibrator only when that improves the Brier score; the
  report has a calibration chart ("when it says 40%, do 40% leave?").
- **A decision rule from the economics, not from 0.5.** An offer costs money and only some offers work, so a customer is
  contacted when `churn chance × save rate × customer value > offer cost`, i.e. above the break-even chance
  `cost ÷ (save rate × value)` (16.7% with the default settings). Change the numbers with `--offer-cost`,
  `--customer-value`, `--save-rate`.
- **Money in the report.** Expected campaign profit for *contact nobody*, *contact everybody*, *use the model* and a
  perfect crystal ball, plus a profit-against-threshold chart.
- **Explanations and actions.** "What drives this risk" (each detail swapped for the typical value), "what could keep this
  customer" (move to a longer contract, add support, switch to automatic payment, fix support problems, price cut) ranked by
  how far each lowers the churn chance, three similar customers, and a what-if tool.
- **Safe by design.** Input is validated with a reason per field, including impossible combinations; saved models are
  verified by SHA-256 before loading; the web server only answers to localhost, limits request size and escapes output.

## Results (synthetic data, 598 unseen test customers, 27.6% churn)

| Model (5-fold CV on training customers) | ROC AUC | PR AUC | Brier (lower is better) |
|---|---|---|---|
| **Logistic Regression** ← chosen | **0.836** | 0.672 | 0.139 |
| Gradient Boosting | 0.825 | 0.652 | 0.144 |
| Random Forest | 0.822 | 0.645 | 0.145 |
| Decision Tree | 0.787 | 0.575 | 0.157 |
| Baseline (everyone stays) | 0.500 | 0.276 | 0.200 |

On the test set: **ROC AUC 0.840**, PR AUC 0.704, Brier 0.136. The best any model could do on this data is
**0.841** (the true rule against luck), so the model is at the ceiling. Training AUC is 0.844: no overfitting.
At the 16.7% contact threshold the model flags 322 of 598 customers, reaches **143 of the 165 who left (87%)**, and 44%
of those flagged really left. The usual 0.5 cut-off would flag only 128 customers and miss 47% of the churners.

| Policy (test customers) | Contacted | Churners reached | Expected profit |
|---|---|---|---|
| Contact nobody | 0 | 0 | ₹0 |
| Contact everybody | 598 | 165 | ₹2.35 lakh |
| **Use the model** | 322 | 143 | **₹3.22 lakh** |
| Perfect foresight (impossible) | 165 | 165 | ₹4.95 lakh |

The first decile of customers (most at risk) churns at 85%, a lift of 3.1×; the top three deciles hold 67% of all churners.
Contract type, billing amounts and tenure drive the prediction. Logistic regression wins because the generating rule is
close to logistic; on messy real data the flexible models usually do better, which is why they stay in the comparison.

## Project structure

```
CustomerChurnPrediction/
├── main.py                  entry point (menu or command line)
├── requirements.txt
├── churnpredict/
│   ├── schema.py            features, valid ranges, spellings, campaign defaults, risk tiers
│   ├── datagen.py           synthetic customers with a known churn rule and injected problems
│   ├── preprocessing.py     loading, cleaning, feature engineering, transformers
│   ├── models.py            candidate models, cross-validation, tuning, calibration choice
│   ├── evaluation.py        metrics, campaign economics, lift, calibration, importance, ceiling
│   ├── training.py          the workflow end to end
│   ├── persistence.py       save / load with SHA-256 checks
│   ├── validators.py        validation of customer records and campaign settings
│   ├── predictor.py         probability, offer decision, factors, actions, what-if, similar customers, batch CSV
│   ├── report.py            text and HTML (SVG charts) reports
│   ├── web.py               local web form and JSON API
│   ├── cli.py               interactive menu
│   └── exceptions.py        custom errors
├── sample_data/             customers_dirty.csv (practice cleaning), customers_to_score.csv (batch example)
├── tests/                   97 automated tests
└── docs/PROJECT_REPORT.md   full project report
```

## How to run

Python 3.10+ is needed.

```bash
cd CustomerChurnPrediction
pip install -r requirements.txt
python main.py --train        # about 20-40 seconds; creates models/
python main.py                # interactive menu
python main.py --serve        # web form at http://127.0.0.1:8030/
```

Other commands:

```bash
python main.py --predict contract="Month-to-month" internet_service=Fiber tenure_months=5 monthly_charges=95 support_calls=4
python main.py --report reports/model_report.html
python main.py --train --offer-cost 800 --customer-value 15000 --save-rate 0.25
python main.py --train --data sample_data/customers_dirty.csv
python main.py --generate sample_data/more_customers.csv --rows 5000
```

JSON API (while the web server runs):

```bash
curl -X POST http://127.0.0.1:8030/api/predict -H "Content-Type: application/json" \
  -d '{"contract":"Month-to-month","internet_service":"Fiber","tenure_months":5,"monthly_charges":95}'
```

Required details: contract, internet service, tenure in months and monthly charges. The rest (total charges, support calls,
late payments, payment method, paperless billing, tech support, online security) fall back to typical values, and the
result says which defaults were used. Total charges default to tenure × monthly charges.

Your own data: a CSV with the columns `tenure_months, monthly_charges, total_charges, support_calls, late_payments,
paperless_billing, tech_support, online_security, contract, internet_service, payment_method, churn` (and optionally
`customer_id`). Open `sample_data/customers_dirty.csv` for an example.

Batch: menu option 3, with a CSV in that format (churn not needed). The output keeps your columns and adds the churn
probability, risk tier, whether to make an offer, the expected gain and a **priority rank** (1 = most valuable offer), so
the retention team can work down the list. Bad rows are reported by line number; good rows are still scored.

## Concepts demonstrated

Binary classification with class imbalance · `Pipeline` and `ColumnTransformer` · imputation, scaling, one-hot encoding ·
feature engineering · stratified cross-validation · hyper-parameter search · ROC and PR curves · probability calibration
(Brier score, reliability chart) · lift and cumulative gains · cost-sensitive decisions and expected value · break-even
analysis · overfitting check · permutation feature importance · explanation by substitution · what-if analysis ·
model persistence with integrity checks · input validation · responsible use of ML about people.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

97 tests (about 1–1.5 minutes) cover the schema helpers, the data generator and its churn rule, file loading, every cleaning
rule, feature engineering, metric maths worked by hand, the campaign economics and threshold, lift and calibration tables,
model quality and reproducibility, persistence and tamper detection, validation (including impossible combinations),
predictions and their direction, retention actions, what-if, batch files and ranks, the web server (HTML and JSON, malformed
requests, escaping) and the menu. A small model is trained once and shared.

## Limitations

- The data is synthetic. Accuracy on a real business would differ and would need real, consented records and a fairness review.
- The save rate and customer value are assumptions: the profit numbers are only as good as those inputs.
- "What could keep this customer" shows patterns in the data, not proven cause and effect. Test offers on a small group first.
- Churn depends on things not in the data (a competitor's deal, a house move), so even a perfect model cannot be perfect.
- Pickle-based model files must only be loaded if you trained them yourself.
