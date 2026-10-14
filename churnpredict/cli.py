"""Interactive menu-driven command-line interface.

All ChurnError subclasses are caught here and shown as friendly messages; unexpected errors are
logged and the menu keeps running.
"""

import logging
from pathlib import Path

from . import datagen, persistence, report, schema as S, training
from . import preprocessing as P
from . import validators as v
from .exceptions import ChurnError, ValidationError
from .predictor import Predictor
from .web import serve

log = logging.getLogger("churnpredict")

MENU = """
============== CUSTOMER CHURN PREDICTION ===================
 PREDICT                            MODEL
  1. Predict one customer            5. Train a new model
  2. What-if: change one detail      6. Model report (accuracy, money)
  3. Score customers from a CSV      7. Compare models
  4. Start the web interface         8. Save an HTML report
 DATA
  9. Create a sample dataset (CSV)  10. Check a dataset's quality
  0. Exit
============================================================"""

# (feature, prompt, hint) in the order they are asked
QUESTIONS = [
    ("contract", "Contract", ", ".join(S.CONTRACTS)),
    ("internet_service", "Internet service", ", ".join(S.INTERNET)),
    ("tenure_months", "Tenure in months", "0-72"),
    ("monthly_charges", "Monthly charges", "10-150"),
    ("total_charges", "Total charges so far", "blank = tenure × monthly"),
    ("support_calls", "Support calls in the last 3 months", "0-20"),
    ("late_payments", "Late payments in the last 12 months", "0-12"),
    ("payment_method", "Payment method", ", ".join(S.PAYMENTS)),
    ("paperless_billing", "Paperless billing? yes/no", ""),
    ("tech_support", "Tech-support add-on? yes/no", ""),
    ("online_security", "Online-security add-on? yes/no", ""),
]
SYNTHETIC_ROWS = 3000
PLAUSIBLE = {"contract": "Month-to-month", "internet_service": "DSL", "tenure_months": 12, "monthly_charges": 50}


def ask(prompt):
    return input(f"  {prompt}: ").strip()


def confirm(prompt, default=False):
    answer = ask(f"{prompt} [{'Y/n' if default else 'y/N'}]").lower()
    return default if not answer else answer in {"y", "yes"}


def _default_text(name):
    default = v.DEFAULTS.get(name)
    if default is None:
        return ""
    if name in S.BINARY:
        return " [yes]" if default else " [no]"
    return f" [{default}]"


class ChurnCLI:
    def __init__(self, model_dir="models", reports_dir="reports"):
        self.model_dir = Path(model_dir)
        self.reports_dir = Path(reports_dir)
        self._predictor = None
        self.actions = {"1": self.predict, "2": self.what_if, "3": self.predict_csv, "4": self.web,
                        "5": self.train, "6": self.model_report, "7": self.compare, "8": self.save_report,
                        "9": self.make_dataset, "10": self.check_quality}

    # ---------------------------------------------------------- plumbing
    def predictor(self):
        if self._predictor is None:
            self._predictor = Predictor.from_directory(self.model_dir)
            for warning in self._predictor.load_warnings:
                print(f"  ! {warning}")
        return self._predictor

    def run(self):
        if not persistence.is_trained(self.model_dir):
            print("\n  No trained model yet.")
            if confirm("Train one now on the built-in sample data (about half a minute)?", default=True):
                self._run_action(self.train_default)
        while True:
            print(MENU)
            choice = ask("Choose an option")
            if choice == "0":
                print("  Goodbye!")
                return
            action = self.actions.get(choice)
            if action is None:
                print("  ! Invalid option, please enter a number from the menu.")
                continue
            self._run_action(action)

    def _run_action(self, action):
        try:
            action()
        except ChurnError as exc:
            log.warning("%s failed: %s", action.__name__, exc)
            print(f"  ! Error: {exc}")
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
        except Exception:  # safety net; the full traceback goes to the log file
            log.exception("Unexpected error in %s", action.__name__)
            print("  ! An unexpected error occurred. See churnpredict.log for details.")

    # ---------------------------------------------------------- prediction
    def ask_customer(self):
        """Ask for each detail, re-asking (up to 3 times) when a value is invalid. Blank = default for optional ones."""
        print("  Describe the customer (press Enter to use the value in brackets):")
        raw = {}
        # Each answer is validated together with earlier answers (and plausible values for the rest), so the
        # message names the field that is wrong, and combinations are checked (e.g. add-ons need internet).
        context = dict(PLAUSIBLE)
        for name, prompt, hint in QUESTIONS:
            shown = _default_text(name)
            for attempt in range(3):
                value = ask(f"{prompt}{shown}" + (f"  ({hint})" if hint else ""))
                if value == "" and name not in v.REQUIRED_INPUT:
                    break   # keep the default
                try:
                    if value == "":
                        raise ValidationError(f"{S.LABELS[name]} is required.")
                    v.parse_features({**context, name: value})
                except ValidationError as exc:
                    print(f"    ! {exc.errors.get(name) or exc}")
                    if attempt == 2:
                        raise ValidationError("Too many invalid answers; prediction cancelled.") from None
                else:
                    raw[name] = context[name] = value
                    break
        return raw

    def show_prediction(self, p, meta=None):
        print(f"\n  Churn chance:   {p.probability * 100:.1f}%   ({p.tier} risk)")
        if p.contact:
            print(f"  Retention offer: WORTH MAKING. Expected gain about {S.format_money(p.expected_gain)} per customer.")
        else:
            print(f"  Retention offer: not needed. It would cost more than it saves ({S.format_money(p.expected_gain)} expected).")
        for warning in p.warnings:
            print(f"  ! {warning}")
        if p.actions:
            print("\n  What could keep this customer (estimates, not guarantees):")
            for a in p.actions:
                print(f"   -{a['reduction_points']:4.0f} points  {a['action']}  "
                      f"({p.probability * 100:.0f}% → {a['new_probability'] * 100:.0f}%)")
        if p.factors:
            print("\n  What drives this risk (compared with a typical customer):")
            top = max(abs(f["effect_points"]) for f in p.factors) or 1
            for f in p.factors:
                sign = "+" if f["effect_points"] >= 0 else "-"
                print(f"   {f['label']:<26} {f['value']:<18} vs {f['typical']:<18} {sign}{abs(f['effect_points']):4.0f} pts  "
                      f"{report.bar(abs(f['effect_points']), top, 12)}")
        if p.similar:
            print("\n  Similar customers in the training data:")
            for c in p.similar:
                tenure = "-" if c["tenure_months"] is None else c["tenure_months"]
                print(f"   {c['contract']:<15} {c['internet_service']:<6} tenure {tenure:>3} months  → {c['outcome']}")

    def predict(self):
        predictor = self.predictor()
        self.show_prediction(predictor.predict(self.ask_customer()))

    def what_if(self):
        predictor = self.predictor()
        raw = self.ask_customer()
        for i, name in enumerate(S.RAW_FEATURES, 1):
            print(f"   {i:>2}. {S.LABELS[name]}")
        choice = ask("Which detail to vary? (number or name)")
        name = S.RAW_FEATURES[int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(S.RAW_FEATURES) else choice
        key = name.strip().lower().replace(" ", "_")
        suggestions = predictor.comparison_values(key) if key in S.RAW_FEATURES else []
        typed = ask(f"Values to try, comma-separated (Enter = {', '.join(map(str, suggestions)) or '?'})")
        values = [x.strip() for x in typed.split(",") if x.strip()] or suggestions
        results = predictor.what_if(raw, name, values)
        print(f"\n  Churn chance of the same customer with a different {S.LABELS[key].lower()}:")
        for value, p in results:
            print(f"   {str(value):<22} {p * 100:5.1f}%  {report.bar(p, 1, 28)}")

    def predict_csv(self):
        predictor = self.predictor()
        print(f"  The CSV needs the columns {', '.join(v.REQUIRED_INPUT)}; other details are optional "
              "(customer_id is kept if present).")
        source = ask("CSV file with customers")
        default = Path("predictions") / f"{Path(source).stem}_scored.csv"
        target = ask(f"Save scores to [{default}]") or default
        ok, errors, summary = predictor.predict_csv(source, target)
        print(f"  Scored {ok} customer(s); saved to {target}.")
        print(f"  High risk {summary['high']} · medium {summary['medium']} · low {summary['low']} · offers worth making "
              f"{summary['offers']} (expected gain {S.format_money(summary['expected_gain'])}).")
        if errors:
            print(f"  {len(errors)} row(s) could not be scored:")
            for line in errors[:15]:
                print(f"    - {line}")

    def web(self):
        port = v.validate_port(ask("Port [8030]") or 8030)
        serve(self.predictor(), port)

    # ---------------------------------------------------------- training and reports
    def train_default(self):
        self._train(datagen.generate(SYNTHETIC_ROWS), "synthetic (generated)", fast=False, tune=True)

    def train(self):
        source = ask("CSV file to train on (Enter = generate the built-in sample data)")
        if source:
            df, label = P.load_dataset(source), Path(source.strip('"')).name
        else:
            rows = v.validate_int(ask(f"Customers to generate [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 500, 30000)
            df, label = datagen.generate(rows), "synthetic (generated)"
        value = v.validate_number(ask(f"Value of a saved customer in rupees [{S.CUSTOMER_VALUE:g}]") or S.CUSTOMER_VALUE,
                                  "Customer value", 1, 10_000_000)
        cost = v.validate_number(ask(f"Cost of one retention offer in rupees [{S.OFFER_COST:g}]") or S.OFFER_COST,
                                 "Offer cost", 1, 10_000_000)
        save = v.validate_number(ask(f"Share of would-be churners an offer saves, 0-1 [{S.SAVE_RATE:g}]") or S.SAVE_RATE,
                                 "Save rate", 0.01, 1)
        fast = confirm("Quick mode (smaller models, faster, a little less accurate)?")
        tune = False if fast else confirm("Tune the best model? (adds a few seconds)", default=True)
        self._train(df, label, fast, tune, value=value, cost=cost, save_rate=save)

    def _train(self, df, label, fast, tune, **economics):
        print()
        result = training.train(df, fast=fast, tune=tune, source=label, progress=lambda m: print(f"   {m}"), **economics)
        persistence.save(result, self.model_dir)
        self._predictor = None
        print("\n  Data cleaning:")
        print("\n".join("  " + line for line in result.cleaning.lines()))
        print()
        print(report.comparison_text(result.metadata))
        print()
        print(report.model_text(result.metadata))
        print(f"\n  Model saved to {self.model_dir}/ — you can now predict.")

    def model_report(self):
        print(report.model_text(self.predictor().meta))

    def compare(self):
        print(report.comparison_text(self.predictor().meta))

    def save_report(self):
        default = self.reports_dir / "model_report.html"
        path = Path(ask(f"File name [{default}]") or default)
        if path.suffix.lower() not in (".html", ".htm"):
            path = path.with_suffix(".html")
        report.save_html(self.predictor().meta, path)
        print(f"  Report saved to {path} — open it in a web browser.")

    # ---------------------------------------------------------- data
    def make_dataset(self):
        rows = v.validate_int(ask(f"Number of customers [{SYNTHETIC_ROWS}]") or SYNTHETIC_ROWS, "Rows", 200, 50000)
        dirty = not ask("Include data problems for practising cleaning? [Y/n]").lower().startswith("n")
        default = Path("sample_data") / ("customers_dirty.csv" if dirty else "customers_clean.csv")
        path = Path(ask(f"File name [{default}]") or default)
        if path.exists():
            raise ValidationError(f"{path} already exists; choose another name.")
        path.parent.mkdir(parents=True, exist_ok=True)
        datagen.generate(rows, dirty=dirty).to_csv(path, index=False)
        print(f"  Saved {rows} customers to {path} (synthetic data).")

    def check_quality(self):
        df = P.load_dataset(ask("CSV file to check"))
        _, cleaning = P.clean(df)
        print("\n".join("  " + line for line in cleaning.lines()))
        if cleaning.missing_before:
            print("  Missing values before cleaning: " + ", ".join(f"{c} {n}" for c, n in cleaning.missing_before.items()))
            print("  (remaining gaps are filled during training using values learned from the training rows only)")
