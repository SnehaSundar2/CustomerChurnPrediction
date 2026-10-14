# Project Report: Customer Churn Prediction

**Program:** Virtual Internship – Skyrovix IT Solutions Pvt. Ltd. (AI & Machine Learning project)
**Project:** Customer Churn Prediction
**Developed and submitted by:** Sneha S (Intern ID: SKX-2026-6425)
**Due date:** 07 November 2026

---

## 1. Introduction

"Churn" is when a customer stops using a service. For a mobile, broadband or streaming company, winning a new
customer costs far more than keeping an existing one, so knowing *who is about to leave* is valuable, but only
if the company can act on it. Calling every customer with a discount wastes money; calling nobody loses customers.

This project builds that system end to end. It cleans messy account records, trains and compares several models,
makes the probabilities trustworthy, and turns them into a **decision**: contact a customer with a retention offer
only when the expected saving beats the cost. It evaluates the result honestly on customers the model has never seen,
in the same terms a business cares about (rupees earned by the campaign), and offers prediction interfaces (a menu, a
command line, a web form, a JSON API and batch files with a ranked call list). Every prediction comes with the
reasons behind it and the actions most likely to keep that customer.

**About the data.** Real customer records are private, so the project generates a realistic but *artificial*
subscription dataset. Each customer's true chance of leaving comes from a known rule (contract, tenure, internet type,
price, support calls, late payments, payment method and add-ons); whether they actually leave is then a random draw. So
even the true rule cannot predict perfectly, and the evaluation can show how close the model gets to the best score
that is possible at all. The results therefore describe the workflow, not real-world accuracy. The model uses no
personal attributes such as gender or age.

## 2. Objectives

1. **Data preprocessing:** load, validate and clean the data, spot impossible values and impossible combinations, and leave a record of every change.
2. **Model training:** compare several algorithms fairly, tune the best one, make its probabilities honest, and avoid leaking test data into training.
3. **Evaluation:** measure ranking quality, probability quality and, above all, what acting on the predictions earns; study where the model is weak.
4. **Prediction interface:** make the model usable through easy interfaces, with strict input validation, explanations and suggested actions.
5. **Quality:** handle errors safely, document the workflow, and verify the behaviour with automated tests.

## 3. Tools and technologies

| Item | Used |
|---|---|
| Language | Python 3 (tested on 3.13) |
| ML libraries | scikit-learn (pipelines, models, cross-validation, tuning, calibration), pandas, numpy, joblib |
| Standard library | `http.server` (web interface), `hashlib` (checksums), `json`, `csv`, `unittest` |
| Output | Terminal reports, self-contained HTML report with SVG charts, JSON API |

## 4. System design

### 4.1 Workflow

1. **Load** the CSV and check that it has the right columns and enough rows.
2. **Clean** the data and record every action.
3. **Split** 80% / 20% into training and test sets with the same churn share in both. The test set is not touched again until the end.
4. **Compare** a baseline and four models by 5-fold stratified cross-validation on the training customers.
5. **Tune** the best model with a randomised search; keep the tuned version only if cross-validation says it is better.
6. **Calibrate:** compare the Brier score of the model with and without a calibration wrapper on out-of-fold predictions; keep the better one.
7. **Decide:** set the contact threshold from the campaign economics, and check on out-of-fold training predictions that the campaign makes money.
8. **Evaluate once** on the test set: ranking, probabilities, lift, profit against simple policies, segments, feature importance.
9. **Save** the model with checksums, then **predict** through any interface.

### 4.2 The data

| Feature | Meaning | Valid range |
|---|---|---|
| `tenure_months` | Months as a customer | 0–72 |
| `monthly_charges`, `total_charges` | Current bill; total billed so far | 10–150; 0–12,000 |
| `support_calls` | Calls to support in the last 3 months | 0–20 |
| `late_payments` | Late payments in the last 12 months | 0–12 |
| `paperless_billing`, `tech_support`, `online_security` | Yes / no | – |
| `contract` | Month-to-month, One year, Two year | – |
| `internet_service` | None, DSL, Fiber | – |
| `payment_method` | Credit card (auto), Bank transfer (auto), UPI autopay, Manual payment | – |
| **`churn`** (target) | Did the customer leave? | Yes / No |

Three features are engineered from the others, never from the target: *average monthly spend* (total ÷ tenure), *price
change* (current bill minus average spend: has the bill gone up?) and *protection services* (tech support + security).

### 4.3 Modules

| Module | Responsibility |
|---|---|
| `schema.py` | Features, valid ranges, spellings, campaign defaults, risk tiers, money formatting |
| `datagen.py` | Synthetic customers with a known churn rule and injected problems |
| `preprocessing.py` | Loading, cleaning with a report, feature engineering, imputing/scaling/encoding |
| `models.py` | Candidate models, cross-validated comparison, tuning, calibration choice |
| `evaluation.py` | Metrics, campaign economics, lift, calibration tables, importance, ceiling |
| `training.py` | The workflow end to end |
| `persistence.py` | Saving and loading with SHA-256 integrity checks |
| `validators.py` | Validation of customer records and campaign settings |
| `predictor.py` | Probability, offer decision, factors, retention actions, what-if, similar customers, batch CSV |
| `report.py` | Text and HTML reports |
| `web.py`, `cli.py`, `main.py` | Web form and API, menu, command line |
| `exceptions.py` | Custom error types |

### 4.4 The pipeline

Every model is wrapped in the same scikit-learn pipeline: **add engineered features → impute missing values (median /
most frequent) → scale numbers and one-hot encode categories → model**. Because the whole chain lives in one pipeline,
anything that learns from data (medians, scaling, categories) is fitted on training rows only.

## 5. Implementation highlights

**Cleaning with an audit trail.** The generator deliberately produces about 10% problem rows. The cleaner found and
handled all of them, and said what it did:

```
3051 rows read → 2986 kept (65 removed)
  • Rows with a missing or unrecognised churn label: 9 — removed
  • Contract spelling standardised: 114 — e.g. 'MTM' → 'Month-to-month'
  • Rows with an unknown or missing contract: 5 — removed
  • Internet service spelling standardised: 130 — e.g. 'fibre' → 'Fiber'
  • Payment method spelling standardised: 127 — e.g. 'UPI' → 'UPI autopay'
  • Blank total charges for brand-new customers (tenure 0): 20 — set to 0
  • Out-of-range tenure_months: 4 — set to missing (valid: 0–72)
  • Out-of-range monthly_charges: 4 — set to missing (valid: 10–150)
  • Out-of-range total_charges: 5 — set to missing (valid: 0–12000)
  • Out-of-range support_calls: 4 — set to missing (valid: 0–20)
  • Out-of-range late_payments: 3 — set to missing (valid: 0–12)
  • Internet add-ons on an account with no internet service: 25 — add-ons set to no
  • Duplicate rows: 45 — removed (kept the first copy)
  • Customer id appearing more than once with different details: 6 — removed (kept the first)
  Kept: 824 churned and 2162 stayed (27.6% churn)
```

Besides single-value problems the cleaner catches *combinations that cannot be true*: total charges that do not fit
tenure × monthly charges, and tech-support or security add-ons on an account with no internet. The same rules protect
the prediction interfaces, which refuse such records with a message naming the field.

**No leakage.** No cleaning rule uses statistics from other rows. Engineered features use only other inputs. The test set
is used once. A test checks that the features can be built from a table without the churn column.

**From a ranking to a decision.** A churn model ranks customers; a business must decide *how many to call*. The decision
is made with simple economics. Making an offer costs ₹600; a saved customer is worth ₹12,000 of kept revenue; 30% of offers
work. An offer therefore pays for itself when `churn chance × 0.30 × 12,000 > 600`, that is, above a **16.7%** churn
chance. With honest probabilities this break-even chance is exactly the profit-maximising threshold. Searching the training
predictions for the best threshold gave a value around 22%, within about 1.5% of the same profit: the profit curve is flat
near its top, so chasing the maximum only chases noise. The break-even rule is therefore used, and the profit curve in the
report shows it sits near the top. On the test customers it earns slightly more than the searched threshold.

**Honest probabilities.** Break-even reasoning only works if "35%" really means 35%. Models are ranked by ROC AUC, but
the chosen model is wrapped in a calibrator only when that lowers the Brier score on out-of-fold predictions (the program
records the comparison). Logistic regression was already well calibrated, so it was left alone; the calibration chart in
the report confirms the dots follow the diagonal.

**Explanations and actions that a retention team can use.** *What drives this risk* swaps each detail for the typical
value (the median or most common category) and shows the change in the churn chance. *What could keep this customer*
tries the actions a company controls (a longer contract, the add-ons, automatic payment, a 10% price cut, fixing repeated
support problems, sorting out late-payment issues) and ranks them by how many points each removes, with the kept revenue.
Actions that make no sense are never offered: no add-ons without internet, no contract upgrade for a two-year customer.
These are patterns in the data, not proven cause and effect, and the interfaces say so.

**A ranked call list.** The batch mode scores a CSV and adds a *priority rank* (1 = the offer with the highest expected
gain) so the team can work down the list until the budget runs out.

**A safe model file.** `metadata.json` stores the SHA-256 of the model and reference files. Loading verifies them first,
so a damaged or modified file is refused with a clear message instead of being unpickled.

## 6. Results

| Model (5-fold CV on training customers) | ROC AUC | PR AUC | Brier |
|---|---|---|---|
| **Logistic Regression** ← chosen | **0.836** | 0.672 | 0.139 |
| Gradient Boosting | 0.825 | 0.652 | 0.144 |
| Random Forest | 0.822 | 0.645 | 0.145 |
| Decision Tree | 0.787 | 0.575 | 0.157 |
| Baseline (everyone stays) | 0.500 | 0.276 | 0.200 |

Tuning logistic regression did not improve cross-validation, so the defaults were kept (the program reports this instead
of hiding it).

**On the 598 unseen test customers (165 churned, 27.6%):** ROC AUC **0.840**, PR AUC 0.704, Brier 0.136. Training AUC is
0.844, so there is no overfitting.

| At the 16.7% contact threshold | Flagged for an offer | Not flagged |
|---|---|---|
| Really left | 143 | 22 (missed) |
| Really stayed | 179 (offer wasted) | 254 |

Recall 86.7%, precision 44.4%. The usual 0.5 cut-off would flag only 128 customers and miss 47% of those who leave.

| Policy (test customers) | Contacted | Churners reached | Expected profit |
|---|---|---|---|
| Contact nobody | 0 | 0 | ₹0 |
| Contact everybody | 598 | 165 | ₹2.35 lakh |
| **Use the model** | 322 | 143 | **₹3.22 lakh** |
| Perfect foresight (impossible) | 165 | 165 | ₹4.95 lakh |

**Interpretation.**
- The model ranks a customer who left above one who stayed 84% of the time. The best any model could do on this data is 84.1% (the true rule against luck), so the model is at the ceiling.
- Following the model's list earns 37% more than contacting everybody and recovers about two thirds of what a perfect crystal ball would earn.
- The most at-risk tenth of customers churns at 85% (a lift of 3.1×); the top three tenths hold 67% of all churners.
- Contract type (importance 0.10), total and monthly charges (0.08, 0.07) and tenure (0.02) drive the prediction.
- By contract, the model predicts 41% churn for month-to-month customers (real 44%), 17% for one-year (real 12%) and 7% for two-year (real 6%): right in order, slightly over-cautious for the loyal groups.
- Logistic regression wins because the generating rule is close to logistic. On messy real data the flexible models usually do better, which is why they stay in the comparison.
- A precision of 44% means more than half of the offers go to customers who would have stayed. That is acceptable only because an offer is cheap (₹600) next to a saved customer (₹12,000); with different numbers the threshold moves, which is why it is a setting.

## 7. Testing

97 automated tests were written and **all pass** (about 1–1.5 minutes). The whole suite was run repeatedly without a
failure. One small model is trained once and shared by the tests.

| # | Test case | Expected | Result |
|---|---|---|---|
| 1 | `MTM`, `1 year`, `fibre`, `no internet service`, `UPI`, `manual`; unknown values | Standardised, or rejected | Pass |
| 2 | `Yes`, `Y`, `1`, `churned`, `retained`, `unsure`, empty | Parsed, or rejected | Pass |
| 3 | Risk tiers at their boundaries; rupee formatting (₹, lakh, crore, negatives) | Correct tier and text | Pass |
| 4 | No gender, age, partner or similar feature | None in the schema | Pass |
| 5 | Generator run twice with the same seed | Identical data; different with another seed | Pass |
| 6 | Clean generated data | No gaps, unique ids, about 26% churn, totals consistent, no add-ons without internet | Pass |
| 7 | The true churn rule | Each factor moves the chance the stated way | Pass |
| 8 | Ceiling: the true rule against real outcomes | ROC AUC between 0.78 and 0.92 (never perfect) | Pass |
| 9 | CSV: missing, empty, wrong columns, 50 rows, not UTF-8; id optional | Specific `DataError` for each | Pass |
| 10 | Blank total (" ") in a file | Survives loading, handled in cleaning | Pass |
| 11 | One data problem at a time (missing/unknown label, missing/unknown contract, duplicate row, repeated id) | Row removed exactly as documented | Pass |
| 12 | Impossible tenure, bill, calls, late payments, total | Value blanked, row kept | Pass |
| 13 | Blank total for tenure 0; total that cannot match tenure × bill; plausible total | 0 / blanked / untouched | Pass |
| 14 | Add-ons on an account with no internet | Set to no and reported | Pass |
| 15 | Spellings, yes/no, labels, unknown payment | Standardised; unclear details become gaps, not deletions | Pass |
| 16 | Cleaning report accounting; input not modified | Rows dropped = rows removed by the listed steps | Pass |
| 17 | Every contract unknown; nobody churned | Refused with a clear reason | Pass |
| 18 | Features built without the label; hand-worked values; tenure 0 | No leakage; no divide-by-zero | Pass |
| 19 | Gaps and an unseen payment method through the pipeline | Finite probabilities, no crash | Pass |
| 20 | Metrics and profit worked by hand | Counts, precision/recall 50%, profit ₹300 | Pass |
| 21 | Threshold on toy data; with honest probabilities; unprofitable campaign | Break-even rule; matches the searched best within 1% profit; nobody contacted | Pass |
| 22 | Policy table, lift, gains and calibration tables | Perfect foresight ≥ model ≥ contact-all; monotone gains; honest bands | Pass |
| 23 | Chosen model vs baseline | AUC above 0.78 vs 0.5; better Brier score | Pass |
| 24 | Churn share equal in train and test; fast same-seed retraining | Within 2 points; identical results | Pass |
| 25 | Precision, recall, PR AUC, Brier on unseen customers | Above the stated floors; no overfitting | Pass |
| 26 | Default 0.5 cut-off | Misses more churners than the chosen threshold | Pass |
| 27 | Campaign settings (value, cost, save rate) | Reach the model; bad values refused | Pass |
| 28 | One byte flipped in `model.joblib`; missing file; bad JSON; no or wrong checksum | Refused with a clear `ModelError` | Pass |
| 29 | At-risk and loyal example customers | High risk / offer vs Low risk / no offer; expected gain sign right | Pass |
| 30 | Direction of each factor (contract, tenure, price, calls, payment, add-ons) | Risk moves the right way | Pass |
| 31 | Add-ons with no internet; total that does not fit tenure; tenure 99 | Rejected with the reason per field | Pass |
| 32 | Retention actions | Ranked, real reductions, sensible for each customer | Pass |
| 33 | Batch CSV with bad contract, add-on without internet, tenure 150, blanks | 3 scored, 3 reported by line number, ranks 1..3, summary adds up | Pass |
| 34 | Web: `"><script>alert(1)</script>` in fields | Escaped; never appears as markup | Pass |
| 35 | Web: wrong content type / bad JSON / 30 KB body / foreign Host header | 415 / 400 / 413 / 403 | Pass |
| 36 | Web: high-risk and low-risk customers, errors, blank total | Right wording; values kept; field named | Pass |
| 37 | Menu: bad answers re-asked; required answer skipped; add-on without internet; batch; training | Friendly output at each step | Pass |

## 8. Challenges and solutions

| Challenge | Solution |
|---|---|
| Real customer data is private | A generator with a known churn rule, so results are reproducible and can be compared with the best possible score |
| Only about a quarter of customers churn | Stratified split and folds; rank models by ROC AUC and PR AUC, not accuracy (the baseline is 72% accurate and useless) |
| A ranking is not a decision | Contact threshold from the economics of an offer; profit compared with contacting nobody, everybody, and a perfect oracle |
| A searched threshold chases noise and contradicted the expected-gain figure | Use the break-even chance (optimal for calibrated probabilities); report the searched value only as a check |
| Probabilities must mean what they say | Calibration compared by Brier score on out-of-fold predictions; calibration chart in the report |
| Impossible records (add-ons without internet, totals that do not fit tenure) | Cleaned in training data and refused in every interface, with a message per field |
| A model should say *why* and *what next* | Per-detail "swap for typical" effects and ranked retention actions that only offer sensible changes |
| Reading the profit chart | The marked point and the table view show the chosen threshold sits near the top of a flat curve |
| Fairness | No personal attributes in the model; errors reported by contract, internet and payment segments |
| Pickled models can be altered or damaged | SHA-256 verified before loading, plus a clear warning to load only your own models |
| An oversized web request could be lost on Windows | The server reads and discards a bounded amount of the body before sending the 413 reply |

## 9. Future enhancements

- Train and compare on a real, consented customer dataset and report both results side by side.
- Run a small controlled trial of the offers to measure the save rate, instead of assuming 30%.
- Predict *who responds to an offer* (uplift modelling), not just who will leave.
- Add survival analysis to predict *when* a customer will leave, not only whether.
- Keep a log of predictions and outcomes and retrain when the data drift.

## 10. Conclusion

The Customer Churn Prediction project meets all its objectives. It cleans messy account data, catching impossible values and
impossible combinations, and records every change. It compares models fairly, keeps the probabilities honest, and turns them into
a decision rule that comes from the economics of a retention campaign. On customers it has never seen it ranks leavers above
stayers 84% of the time, which is the best the data allows, and following its list earns 37% more than contacting everybody. Each
prediction comes with the factors behind it, the actions most likely to keep the customer, and similar customers, through a menu, a
command line, a web form, a JSON API and ranked batch files. Inputs are validated strictly, errors are handled clearly, saved models
are integrity-checked, and 97 automated tests verify the behaviour.

## 11. How to run

```bash
cd CustomerChurnPrediction
pip install -r requirements.txt
python main.py --train
python main.py
python main.py --serve
python -m unittest discover -s tests -t . -v
```
