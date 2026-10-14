"""Input validation, model persistence, predictions, explanations, retention actions, what-if and batch files."""

import json

import pandas as pd

from churnpredict import persistence, schema as S, validators as v
from churnpredict.exceptions import DataError, ModelError, NotTrainedError, ValidationError
from churnpredict.predictor import Predictor, require_predictor
from tests.helpers import AT_RISK, LOYAL, TempDirTest, shared


class TestValidation(TempDirTest):
    def test_accepted_formats(self):
        self.assertEqual(v.parse_number("monthly_charges", "Rs 95"), 95.0)
        self.assertEqual(v.parse_number("monthly_charges", "₹ 82.5"), 82.5)
        self.assertEqual(v.parse_number("tenure_months", "14 months"), 14)
        self.assertEqual(v.parse_number("support_calls", "5 calls"), 5)
        self.assertEqual(v.parse_number("tenure_months", 0), 0)                    # the edges of the range are fine
        self.assertEqual(v.parse_number("tenure_months", 72), 72)
        self.assertEqual(v.parse_number("tenure_months", 12.0), 12)                # a whole number written with a decimal point

    def test_rejected_values_explain_why(self):
        cases = [("tenure_months", 73, "between 0 months and 72 months"), ("tenure_months", -1, "between"),
                 ("tenure_months", 2.5, "whole number"), ("monthly_charges", 5, "between 10 and 150"),
                 ("monthly_charges", "abc", "must be a number"), ("monthly_charges", "", "must be a number"),
                 ("monthly_charges", float("nan"), "normal number"), ("monthly_charges", float("inf"), "normal number"),
                 ("support_calls", True, "must be a number"), ("support_calls", 21, "between 0 and 20"),
                 ("late_payments", "1e9", "must be a number")]
        for feature, value, hint in cases:
            with self.assertRaises(ValueError, msg=repr(value)) as ctx:
                v.parse_number(feature, value)
            self.assertIn(hint, str(ctx.exception))

    def test_category_aliases_and_suggestions(self):
        self.assertEqual(v.parse_category("contract", " mtm "), "Month-to-month")
        self.assertEqual(v.parse_category("payment_method", "UPI"), "UPI autopay")
        with self.assertRaises(ValueError) as ctx:
            v.parse_category("internet_service", "Fibr")
        self.assertIn("Did you mean 'Fiber'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            v.parse_category("contract", "Prepaid")
        self.assertIn("Choose one of: Month-to-month, One year, Two year", str(ctx.exception))

    def test_defaults_and_derived_total_are_reported(self):
        required = {k: AT_RISK[k] for k in v.REQUIRED_INPUT}
        features, assumed = v.parse_features(required)
        self.assertEqual(set(assumed), set(v.DEFAULTS) | {"total_charges"})
        self.assertEqual(features["total_charges"], 475.0)                         # 5 months × 95
        self.assertEqual(set(features), set(S.RAW_FEATURES))
        full, assumed = v.parse_features({**AT_RISK, "total_charges": 480})
        self.assertEqual(assumed, {})
        self.assertEqual(full["total_charges"], 480.0)

    def test_keys_are_forgiving(self):
        features, _ = v.parse_features({"Tenure Months": "9", "contract": "two year", "Monthly-Charges": 60,
                                        "INTERNET_SERVICE": "dsl", "customer_id": "CU1"})      # an id is allowed and ignored
        self.assertEqual((features["tenure_months"], features["contract"], features["internet_service"]), (9, "Two year", "DSL"))
        self.assertNotIn("customer_id", features)

    def test_all_problems_are_reported_together(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({"contract": "Prepaid", "tenure_months": 99, "monthly_charges": "lots", "bogus": 1})
        errors = ctx.exception.errors
        self.assertTrue({"contract", "tenure_months", "monthly_charges", "internet_service", "_"} <= set(errors))
        with self.assertRaises(ValidationError):
            v.parse_features("not a record")

    def test_combinations_that_cannot_be_true_are_refused(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({**AT_RISK, "internet_service": "None", "tech_support": "yes"})
        self.assertIn("needs an internet service", ctx.exception.errors["tech_support"])
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({**AT_RISK, "tenure_months": 24, "monthly_charges": 50, "total_charges": 100})
        self.assertIn("do not fit", ctx.exception.errors["total_charges"])
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({**AT_RISK, "tenure_months": 1, "monthly_charges": 50, "total_charges": 5000})
        self.assertIn("too high", ctx.exception.errors["total_charges"])
        features, _ = v.parse_features({**AT_RISK, "tenure_months": 0, "total_charges": 0})
        self.assertEqual(features["total_charges"], 0)                             # a brand-new customer has paid nothing

    def test_blank_values_and_yes_no(self):
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({**AT_RISK, "tenure_months": "  ", "contract": None})
        self.assertEqual(set(ctx.exception.errors), {"tenure_months", "contract"})
        features, _ = v.parse_features({**AT_RISK, "tech_support": "Yes", "paperless_billing": 0})
        self.assertEqual((features["tech_support"], features["paperless_billing"]), (1, 0))
        with self.assertRaises(ValidationError) as ctx:
            v.parse_features({**AT_RISK, "tech_support": "sometimes"})
        self.assertIn("yes or no", ctx.exception.errors["tech_support"])

    def test_settings_validation(self):
        self.assertEqual(v.validate_economics(12000, 600, 0.3), (12000.0, 600.0, 0.3))
        for args in ((0, 600, 0.3), (12000, -1, 0.3), (12000, 600, 0), (12000, 600, 1.2), ("x", 600, 0.3)):
            with self.assertRaises(ValidationError, msg=str(args)):
                v.validate_economics(*args)
        self.assertEqual(v.validate_port("8030"), 8030)
        self.assertEqual(v.validate_int("5", "Rows", 1, 10), 5)
        self.assertEqual(v.validate_number("0.25", "Rate", 0, 1), 0.25)
        for bad in ("abc", "80", "70000", ""):
            with self.assertRaises(ValidationError):
                v.validate_port(bad)
        for bad in ("x", "2", "-1", "nan", ""):
            with self.assertRaises(ValidationError):
                v.validate_number(bad, "Rate", 0, 1)


class TestPersistence(TempDirTest):
    def test_round_trip_gives_identical_predictions(self):
        result, model_dir, _ = shared()
        a = Predictor(persistence.load(model_dir)).predict(AT_RISK)
        b = Predictor(persistence.load(model_dir)).predict(AT_RISK)
        self.assertEqual(a.probability, b.probability)
        features, _ = v.parse_features(AT_RISK)
        original = float(result.pipeline.predict_proba(pd.DataFrame([features])[S.RAW_FEATURES])[0, 1])
        self.assertAlmostEqual(a.probability, original, places=9)
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(set(meta["files"]), {"model.joblib", "reference.csv"})
        self.assertEqual(len(meta["files"]["model.joblib"]), 64)
        self.assertEqual([p.name for p in model_dir.glob("*.tmp")], [])            # atomic saves leave no temp files

    def test_no_model_gives_instructions(self):
        for call in (lambda: persistence.load(self.dir / "empty"), lambda: require_predictor(self.dir / "empty")):
            with self.assertRaises(NotTrainedError) as ctx:
                call()
            self.assertIn("Train one first", str(ctx.exception))

    def test_tampering_and_damage_are_detected(self):
        model = self.copy_model()
        data = bytearray((model / "model.joblib").read_bytes())
        data[len(data) // 2] ^= 0x01
        (model / "model.joblib").write_bytes(bytes(data))
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("checksum", str(ctx.exception))

        model = self.copy_model()
        (model / "reference.csv").unlink()
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("missing", str(ctx.exception))

        model = self.copy_model()
        meta_path = model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["files"]["model.joblib"] = "0" * 64
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaises(ModelError):
            persistence.load(model)

        meta_path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("damaged", str(ctx.exception))

        meta_path.write_text(json.dumps({"model": "x"}), encoding="utf-8")
        with self.assertRaises(ModelError) as ctx:
            persistence.load(model)
        self.assertIn("no checksums", str(ctx.exception))

    def test_library_version_mismatch_warns(self):
        model = self.copy_model()
        meta_path = model / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["environment"]["sklearn"] = "0.20.0"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        loaded = persistence.load(model)
        self.assertTrue(any("0.20.0" in w for w in loaded.warnings))


class TestPredictor(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor.from_directory(shared()[1])

    def p(self, **changes):
        return self.predictor.predict({**AT_RISK, **changes}, explain=False, similar=False).probability

    def test_prediction_structure_and_json(self):
        pred = self.predictor.predict(AT_RISK)
        self.assertTrue(0 <= pred.probability <= 1)
        self.assertEqual(pred.tier, S.risk_tier(pred.probability, self.predictor.threshold))
        self.assertEqual(pred.contact, pred.probability >= self.predictor.threshold)
        data = pred.to_dict()
        for key in ("churn_probability", "risk_tier", "make_offer", "expected_gain", "assumed_defaults", "warnings",
                    "factors", "retention_actions", "similar_customers"):
            self.assertIn(key, data)
        json.dumps(data)                                                           # everything can be sent as JSON

    def test_the_two_example_customers_get_opposite_verdicts(self):
        risky, loyal = self.predictor.predict(AT_RISK), self.predictor.predict(LOYAL)
        self.assertGreater(risky.probability, 0.7)
        self.assertEqual((risky.tier, risky.contact), ("High", True))
        self.assertLess(loyal.probability, 0.1)
        self.assertEqual((loyal.tier, loyal.contact), ("Low", False))
        self.assertGreater(risky.expected_gain, 0)
        self.assertLess(loyal.expected_gain, 0)                                    # an offer to a loyal customer loses money

    def test_expected_gain_formula(self):
        eco = self.predictor.economics
        for features in (AT_RISK, LOYAL):
            pred = self.predictor.predict(features, explain=False, similar=False)
            self.assertAlmostEqual(pred.expected_gain, pred.probability * eco["save_rate"] * eco["value"] - eco["cost"])
        at_break_even = self.predictor.gain(eco["break_even"])
        self.assertAlmostEqual(at_break_even, 0.0, places=6)                       # exactly the break-even chance earns nothing

    def test_risk_moves_in_the_right_direction(self):
        self.assertGreater(self.p(contract="Month-to-month"), self.p(contract="One year"))
        self.assertGreater(self.p(contract="One year"), self.p(contract="Two year"))
        self.assertGreater(self.p(tenure_months=2, total_charges=190), self.p(tenure_months=60, total_charges=5700))
        self.assertGreater(self.p(monthly_charges=130, total_charges=650), self.p(monthly_charges=40, total_charges=200))
        self.assertGreater(self.p(support_calls=8), self.p(support_calls=0))
        self.assertGreater(self.p(payment_method="Manual payment"), self.p(payment_method="UPI autopay"))
        self.assertGreater(self.p(tech_support="no"), self.p(tech_support="yes"))
        self.assertGreater(self.p(online_security="no"), self.p(online_security="yes"))

    def test_assumed_defaults_are_reported(self):
        pred = self.predictor.predict({k: AT_RISK[k] for k in v.REQUIRED_INPUT})
        self.assertTrue(any("Assumed for missing details" in w for w in pred.warnings))
        self.assertIn("total_charges", pred.assumed)

    def test_unusual_values_warn(self):
        pred = self.predictor.predict({**AT_RISK, "monthly_charges": 149, "total_charges": 745})   # allowed, but far above the data
        self.assertTrue(any("outside the range the model learned" in w for w in pred.warnings))
        self.assertEqual([w for w in self.predictor.predict(LOYAL).warnings if "outside" in w], [])

    def test_factors_are_sorted_and_labelled(self):
        pred = self.predictor.predict(AT_RISK)
        effects = [abs(f["effect_points"]) for f in pred.factors]
        self.assertEqual(effects, sorted(effects, reverse=True))
        self.assertLessEqual(len(pred.factors), 6)
        by_feature = {f["feature"]: f for f in pred.factors}
        self.assertNotIn("contract", by_feature)                                  # month-to-month IS the typical contract
        self.assertGreater(by_feature["monthly_charges"]["effect_points"], 5)     # an expensive plan
        self.assertGreater(by_feature["support_calls"]["effect_points"], 0)       # four calls instead of one raise the risk
        loyal = {f["feature"]: f for f in self.predictor.predict(LOYAL).factors}
        self.assertLess(loyal["contract"]["effect_points"], 0)                    # a two-year contract protects
        typical = self.predictor.meta["typical"]
        average = {k: (("yes" if typical[k] else "no") if k in S.BINARY else (round(typical[k]) if S.NUMERIC.get(k, ("",))[0] == "int" else typical[k]))
                   for k in S.RAW_FEATURES}
        self.assertEqual(self.predictor.predict(average).factors, [])             # a typical customer has nothing to explain

    def test_retention_actions_are_ranked_and_real(self):
        pred = self.predictor.predict(AT_RISK)
        self.assertTrue(pred.actions)
        drops = [a["reduction_points"] for a in pred.actions]
        self.assertEqual(drops, sorted(drops, reverse=True))
        self.assertLessEqual(len(pred.actions), 4)
        self.assertTrue(all(a["new_probability"] < pred.probability for a in pred.actions))
        self.assertTrue(any("contract" in a["action"].lower() for a in pred.actions))
        top = pred.actions[0]
        self.assertAlmostEqual(top["retained_value"], (pred.probability - top["new_probability"]) * 12000, delta=2)
        self.assertAlmostEqual(top["reduction_points"], 100 * (pred.probability - top["new_probability"]), delta=0.06)

    def test_actions_only_offer_what_makes_sense(self):
        loyal = self.predictor.predict(LOYAL)
        self.assertFalse(any("contract" in a["action"].lower() for a in loyal.actions))        # already on two years
        self.assertFalse(any("add-on" in a["action"].lower() for a in loyal.actions))          # already has both
        no_net = self.predictor.predict({**AT_RISK, "internet_service": "None", "tech_support": "no", "monthly_charges": 20,
                                         "total_charges": 100, "payment_method": "Bank transfer (auto)"})
        self.assertFalse(any("add-on" in a["action"].lower() for a in no_net.actions))         # no internet, no internet add-ons
        self.assertFalse(any("automatic payment" in a["action"].lower() for a in no_net.actions))

    def test_similar_customers(self):
        pred = self.predictor.predict(AT_RISK)
        self.assertEqual(len(pred.similar), 3)
        self.assertTrue(all(c["outcome"] in {"Churned", "Stayed"} for c in pred.similar))
        self.assertTrue(all(c["contract"] == "Month-to-month" for c in pred.similar))
        self.assertGreaterEqual(sum(c["outcome"] == "Churned" for c in pred.similar), 2)       # look-alikes mostly left

    def test_what_if(self):
        results = self.predictor.what_if(AT_RISK, "contract", ["Month-to-month", "One year", "two year"])
        self.assertEqual([value for value, _ in results], ["Month-to-month", "One year", "Two year"])
        scores = [p for _, p in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        support = self.predictor.what_if(AT_RISK, "Support Calls", [0, 4, 8])
        self.assertEqual([p for _, p in support], sorted(p for _, p in support))
        with self.assertRaises(ValidationError) as ctx:
            self.predictor.what_if(AT_RISK, "tenure_months", [10, 500])
        self.assertIn("500", str(ctx.exception))
        with self.assertRaises(ValidationError):
            self.predictor.what_if(AT_RISK, "favourite_colour", [1])

    def test_comparison_values(self):
        self.assertEqual(self.predictor.comparison_values("contract"), S.CONTRACTS)
        self.assertEqual(self.predictor.comparison_values("tech_support"), [0, 1])
        values = self.predictor.comparison_values("tenure_months")
        self.assertEqual(values, sorted(values))
        self.assertTrue(all(isinstance(x, int) for x in values))

    def test_invalid_input_never_reaches_the_model(self):
        for bad in ({**AT_RISK, "tenure_months": 200}, {**AT_RISK, "contract": "Prepaid"},
                    {**AT_RISK, "monthly_charges": float("nan")}, {"contract": "Two year"},
                    {**AT_RISK, "internet_service": "None", "online_security": "yes"}):
            with self.assertRaises(ValidationError):
                self.predictor.predict(bad)


class TestBatch(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor.from_directory(shared()[1])

    def setUp(self):
        super().setUp()
        self.source = self.dir / "customers.csv"
        pd.DataFrame([
            {"Customer ID": "A1", "Contract": "Month-to-month", "Internet Service": "Fiber", "Tenure Months": 5, "Monthly Charges": 95, "Support Calls": 4, "Payment Method": "manual"},
            {"Customer ID": "A2", "Contract": "Two year", "Internet Service": "DSL", "Tenure Months": 40, "Monthly Charges": 52, "Support Calls": 0, "Payment Method": "UPI"},
            {"Customer ID": "A3", "Contract": "Prepaid", "Internet Service": "DSL", "Tenure Months": 12, "Monthly Charges": 50, "Support Calls": 1, "Payment Method": "UPI"},
            {"Customer ID": "A4", "Contract": "One year", "Internet Service": "Fiber", "Tenure Months": 14, "Monthly Charges": 85, "Support Calls": "", "Payment Method": ""},
            {"Customer ID": "A5", "Contract": "Month-to-month", "Internet Service": "None", "Tenure Months": 8, "Monthly Charges": 20, "Support Calls": 1, "Payment Method": "card", "Tech Support": "yes"},
            {"Customer ID": "A6", "Contract": "Month-to-month", "Internet Service": "DSL", "Tenure Months": 150, "Monthly Charges": 45, "Support Calls": 1, "Payment Method": "card"},
        ]).to_csv(self.source, index=False)

    def test_good_rows_are_scored_and_bad_rows_reported(self):
        out = self.dir / "result.csv"
        ok, errors, summary = self.predictor.predict_csv(self.source, out)
        self.assertEqual(ok, 2 + 1)                                                # A1, A2, A4 are fine
        self.assertEqual([e.split(":")[0] for e in errors], ["Line 4", "Line 6", "Line 7"])
        frame = pd.read_csv(out, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        self.assertEqual(len(frame), 6)
        self.assertEqual(frame["customer_id"].tolist(), ["A1", "A2", "A3", "A4", "A5", "A6"])      # original columns and order kept
        for col in ("churn_probability", "risk_tier", "make_offer", "expected_gain", "priority_rank", "status"):
            self.assertIn(col, frame.columns)
        self.assertEqual(frame["risk_tier"][0], "High")
        self.assertEqual(frame["make_offer"][0], "yes")
        self.assertEqual(frame["risk_tier"][1], "Low")
        self.assertEqual(frame["make_offer"][1], "no")
        self.assertEqual(frame["churn_probability"][2], "")                        # nothing invented for a bad row
        self.assertTrue(frame["status"][2].startswith("error:"))
        self.assertIn("defaults used", frame["status"][3])
        self.assertAlmostEqual(float(frame["churn_probability"][0]), self.predictor.predict(
            {"contract": "Month-to-month", "internet_service": "Fiber", "tenure_months": 5, "monthly_charges": 95,
             "support_calls": 4, "payment_method": "manual"}, explain=False, similar=False).probability, places=3)

    def test_priority_ranks_and_summary(self):
        out = self.dir / "result.csv"
        ok, _, summary = self.predictor.predict_csv(self.source, out)
        frame = pd.read_csv(out, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        ranked = frame[frame["priority_rank"] != ""]
        self.assertEqual(sorted(int(r) for r in ranked["priority_rank"]), list(range(1, ok + 1)))
        gains = [float(g) for g in ranked.sort_values("priority_rank", key=lambda s: s.astype(int))["expected_gain"]]
        self.assertEqual(gains, sorted(gains, reverse=True))                       # rank 1 is the most valuable offer
        self.assertEqual(summary["scored"], ok)
        self.assertEqual(summary["high"] + summary["medium"] + summary["low"], ok)
        self.assertEqual(summary["offers"], int((frame["make_offer"] == "yes").sum()))
        self.assertGreater(summary["expected_gain"], 0)

    def test_file_level_problems(self):
        out = self.dir / "result.csv"
        with self.assertRaises(DataError):
            self.predictor.predict_csv(self.dir / "missing.csv", out)
        no_cols = self.dir / "bad.csv"
        no_cols.write_text("name,age\nAsha,20\n", encoding="utf-8")
        with self.assertRaises(DataError) as ctx:
            self.predictor.predict_csv(no_cols, out)
        self.assertIn("missing required column", str(ctx.exception))
        empty = self.dir / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaises(DataError):
            self.predictor.predict_csv(empty, out)
        self.predictor.predict_csv(self.source, out)
        with self.assertRaises(DataError) as ctx:
            self.predictor.predict_csv(self.source, out)                            # never overwrite
        self.assertIn("already exists", str(ctx.exception))
