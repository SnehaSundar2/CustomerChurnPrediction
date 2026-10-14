"""Schema helpers, synthetic data, dataset loading, cleaning and feature engineering."""

import numpy as np
import pandas as pd

from churnpredict import datagen, evaluation as E, models as M, preprocessing as P, schema as S
from churnpredict.exceptions import DataError
from tests.helpers import TempDirTest


class TestSchemaHelpers(TempDirTest):
    def test_categories_and_aliases(self):
        cases = [("contract", "MTM", "Month-to-month"), ("contract", " month to month ", "Month-to-month"),
                 ("contract", "1 year", "One year"), ("contract", "TWO_YEAR", "Two year"),
                 ("internet_service", "fibre", "Fiber"), ("internet_service", "no internet service", "None"),
                 ("internet_service", "NONE", "None"), ("payment_method", "Credit Card", "Credit card (auto)"),
                 ("payment_method", "UPI", "UPI autopay"), ("payment_method", "manual", "Manual payment")]
        for feature, raw, expected in cases:
            self.assertEqual(S.canonical_category(feature, raw), expected, raw)
        for raw in ("Prepaid", "", None, float("nan")):
            self.assertIsNone(S.canonical_category("contract", raw), repr(raw))

    def test_yes_no_and_labels(self):
        self.assertEqual([S.parse_yes_no(x) for x in ("Yes", "n", True, 0, "1", "TRUE")], [1, 0, 1, 0, 1, 1])
        self.assertIsNone(S.parse_yes_no("maybe"))
        yes = ["Yes", "yes ", "Y", 1, "1", 1.0, "True", "churned", True]
        no = ["No", " no", "N", 0, "0", 0.0, "False", "retained", False]
        self.assertEqual({S.canonical_label(x) for x in yes}, {1})
        self.assertEqual({S.canonical_label(x) for x in no}, {0})
        for bad in ("unsure", "", None, float("nan"), 2):
            self.assertIsNone(S.canonical_label(bad), repr(bad))

    def test_risk_tiers_and_money(self):
        self.assertEqual([S.risk_tier(p, 0.2) for p in (0.5, 0.2, 0.19, 0.1, 0.09, 0.0)],
                         ["High", "High", "Medium", "Medium", "Low", "Low"])
        self.assertEqual(S.risk_tier(0.9, 1.0000001), "Medium")                  # "contact nobody" leaves no High tier
        self.assertEqual([S.format_money(x) for x in (600, 12000, 315000, 12_500_000, -2400, -250_000)],
                         ["₹600", "₹12,000", "₹3.15 lakh", "₹1.25 crore", "-₹2,400", "-₹2.50 lakh"])

    def test_no_personal_attributes_are_used(self):
        forbidden = {"gender", "age", "senior", "marital", "partner", "dependents", "religion", "name"}
        self.assertFalse(any(word in feature for feature in S.RAW_FEATURES for word in forbidden))


class TestDatagen(TempDirTest):
    def test_reproducible_and_dirty(self):
        a, b = datagen.generate(500, seed=1), datagen.generate(500, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertFalse(a.equals(datagen.generate(500, seed=2)))
        self.assertGreater(a.isna().sum().sum(), 0)
        self.assertGreater(a.duplicated().sum(), 0)

    def test_clean_data_is_tidy_and_consistent(self):
        clean = datagen.generate(4000, dirty=False)
        self.assertEqual(clean.isna().sum().sum(), 0)
        self.assertEqual(clean[S.ID].nunique(), 4000)
        self.assertAlmostEqual((clean[S.TARGET] == "Yes").mean(), 0.26, delta=0.05)
        self.assertEqual(set(clean[S.TARGET]), {"Yes", "No"})
        self.assertFalse(((clean["internet_service"] == "None") & ((clean["tech_support"] == 1) | (clean["online_security"] == 1))).any())
        ratio = clean["total_charges"] / (clean["tenure_months"] * clean["monthly_charges"]).replace(0, np.nan)
        self.assertTrue(ratio.dropna().between(*S.TOTAL_RATIO).all())
        self.assertTrue((clean.loc[clean["tenure_months"] == 0, "total_charges"] == 0).all())
        for col, (_, lo, hi, _) in S.NUMERIC.items():
            self.assertTrue(clean[col].between(lo, hi).all(), col)

    def test_the_churn_rule_is_sensible(self):
        def p(**kw):
            base = dict(tenure=[12], monthly=[70.0], support_calls=[1], late=[0], contract=["Month-to-month"],
                        internet=["DSL"], payment=["Bank transfer (auto)"], paperless=[0], tech=[0], security=[0])
            base.update(kw)
            return float(datagen.churn_probability(**base)[0])
        self.assertGreater(p(contract=["Month-to-month"]), p(contract=["One year"]))
        self.assertGreater(p(contract=["One year"]), p(contract=["Two year"]))
        self.assertGreater(p(tenure=[1]), p(tenure=[60]))
        self.assertGreater(p(internet=["Fiber"]), p(internet=["DSL"]))
        self.assertGreater(p(monthly=[110.0]), p(monthly=[40.0]))
        self.assertGreater(p(support_calls=[6]), p(support_calls=[0]))
        self.assertGreater(p(late=[4]), p(late=[0]))
        self.assertGreater(p(payment=["Manual payment"]), p(payment=["UPI autopay"]))
        self.assertGreater(p(tech=[0]), p(tech=[1]))
        self.assertGreater(p(security=[0]), p(security=[1]))
        self.assertTrue(0 < p(tenure=[0], support_calls=[20]) < 1)

    def test_even_the_true_rule_cannot_be_perfect(self):
        ceiling = E.roc_curve_ceiling(datagen.generate(3000, dirty=False))
        self.assertGreater(ceiling["roc_auc"], 0.78)
        self.assertLess(ceiling["roc_auc"], 0.92)                                 # luck decides who really leaves

    def test_sample_data_is_cleanable(self):
        cleaned, report = P.clean(datagen.generate(3000))
        self.assertGreater(report.rows_dropped, 30)
        self.assertLess(report.rows_dropped, 300)
        self.assertEqual(cleaned.duplicated().sum(), 0)


class TestLoading(TempDirTest):
    def write(self, name, text, binary=False):
        path = self.dir / name
        path.write_bytes(text) if binary else path.write_text(text, encoding="utf-8")
        return path

    def test_good_file_header_normalisation_and_optional_id(self):
        df = datagen.generate(300, dirty=False)
        df.columns = [c.replace("_", " ").title() + " " for c in df.columns]      # "Tenure Months ", "Churn " ...
        path = self.dir / "ok.csv"
        df.to_csv(path, index=False)
        loaded = P.load_dataset(path)
        self.assertEqual(list(loaded.columns), [S.ID] + S.REQUIRED_COLUMNS)
        self.assertEqual(len(loaded), 300)
        no_id = datagen.generate(300, dirty=False).drop(columns=[S.ID])
        path = self.dir / "noid.csv"
        no_id.to_csv(path, index=False)
        self.assertEqual(list(P.load_dataset(path).columns), S.REQUIRED_COLUMNS)

    def test_problem_files_are_explained(self):
        cases = {
            "missing.csv": (None, "not found"),
            "empty.csv": ("", "is empty"),
            "cols.csv": ("name,age\nAsha,20\n", "missing column(s)"),
            "small.csv": (datagen.generate(50, dirty=False).to_csv(index=False), "only 50 rows"),
        }
        for name, (text, hint) in cases.items():
            path = self.dir / name if text is None else self.write(name, text)
            with self.assertRaises(DataError, msg=name) as ctx:
                P.load_dataset(path)
            self.assertIn(hint, str(ctx.exception))
        latin = self.write("latin.csv", "contract,churn\nZürich,Yes\n".encode("latin-1"), binary=True)
        with self.assertRaises(DataError) as ctx:
            P.load_dataset(latin)
        self.assertIn("UTF-8", str(ctx.exception))

    def test_blank_total_charges_survive_loading(self):
        df = datagen.generate(300, dirty=False)
        df["total_charges"] = df["total_charges"].astype(object)
        df.loc[0, "total_charges"] = " "
        path = self.dir / "blank.csv"
        df.to_csv(path, index=False)
        cleaned, report = P.clean(P.load_dataset(path))
        self.assertGreaterEqual(len(cleaned), 295)


class TestCleaning(TempDirTest):
    def setUp(self):
        super().setUp()
        self.base = datagen.generate(600, dirty=False, seed=5)

    def with_row(self, **changes):
        """The data plus a copy of row 0 (with a new customer id) changed as given."""
        row = self.base.iloc[[0]].copy()
        row[S.ID] = "CU-NEW"
        for key, value in changes.items():
            row[key] = value
        return pd.concat([self.base.astype(object), row.astype(object)], ignore_index=True)

    def last(self, df):
        out, report = P.clean(df)
        return out, report

    def test_clean_data_passes_through_untouched(self):
        out, report = P.clean(self.base)
        self.assertEqual((report.rows_in, report.rows_out), (600, 600))
        self.assertTrue(all("standardised" in label for label, _, _ in report.steps) or report.steps == [])

    def test_rows_that_cannot_be_used_are_removed(self):
        for label, df in {
            "missing label": self.with_row(churn=np.nan), "unknown label": self.with_row(churn="unsure"),
            "missing contract": self.with_row(contract=np.nan), "unknown contract": self.with_row(contract="Prepaid"),
            "exact duplicate": pd.concat([self.base, self.base.iloc[[3]]], ignore_index=True),
            "id repeated, details differ": self.with_row(**{S.ID: self.base[S.ID].iloc[1], "tenure_months": 71}),
        }.items():
            out, report = P.clean(df)
            expected_dropped = 1
            self.assertEqual(report.rows_dropped, expected_dropped, label)
            self.assertEqual(len(out), len(self.base), label)

    def test_impossible_values_are_blanked_and_the_row_is_kept(self):
        for column, value in {"tenure_months": -3, "monthly_charges": 900, "support_calls": 999, "late_payments": -2,
                              "total_charges": 50000}.items():
            out, report = P.clean(self.with_row(**{column: value, "tenure_months": 12, "monthly_charges": 60}
                                                if column == "total_charges" else {column: value}))
            self.assertEqual(len(out), len(self.base) + 1, column)
            self.assertEqual(out[column].isna().sum(), 1, column)

    def test_total_charges_rules(self):
        out, _ = P.clean(self.with_row(tenure_months=0, total_charges=" "))
        self.assertEqual(out["total_charges"].iloc[-1], 0.0)                       # a brand-new customer has paid nothing
        out, report = P.clean(self.with_row(tenure_months=24, monthly_charges=50, total_charges=20))
        self.assertTrue(pd.isna(out["total_charges"].iloc[-1]))                    # 24 months at 50 cannot total 20
        self.assertTrue(any("cannot match" in label for label, _, _ in report.steps))
        out, _ = P.clean(self.with_row(tenure_months=24, monthly_charges=50, total_charges=1250))
        self.assertEqual(out["total_charges"].iloc[-1], 1250)                      # a plausible total is untouched

    def test_addons_without_internet_are_set_to_no(self):
        out, report = P.clean(self.with_row(internet_service="None", tech_support=1, online_security=1))
        self.assertEqual((out["tech_support"].iloc[-1], out["online_security"].iloc[-1]), (0, 0))
        self.assertTrue(any("no internet service" in label for label, _, _ in report.steps))

    def test_spellings_yes_no_and_labels_are_standardised(self):
        df = self.with_row(contract="MTM", internet_service="fibre", payment_method="Credit Card", paperless_billing="Y",
                           tech_support="no", churn="churned")
        out, report = P.clean(df)
        row = out.iloc[-1]
        self.assertEqual((row["contract"], row["internet_service"], row["payment_method"]),
                         ("Month-to-month", "Fiber", "Credit card (auto)"))
        self.assertEqual((row["paperless_billing"], row["tech_support"], row["churn"]), (1, 0, 1))
        self.assertEqual(out["churn"].dtype, np.dtype("int64"))
        out, _ = P.clean(self.with_row(payment_method="Barter", paperless_billing="maybe"))
        self.assertTrue(pd.isna(out["payment_method"].iloc[-1]) and pd.isna(out["paperless_billing"].iloc[-1]))
        self.assertEqual(len(out), len(self.base) + 1)                             # unclear details are gaps, not deletions

    def test_input_is_not_modified_and_accounting_adds_up(self):
        raw = datagen.generate(1500)
        before = raw.copy()
        out, report = P.clean(raw)
        pd.testing.assert_frame_equal(raw, before)
        removed = sum(count for _, count, outcome in report.steps if outcome.startswith("removed"))
        self.assertEqual(report.rows_dropped, removed)
        self.assertEqual(report.rows_out, len(out))
        self.assertEqual(sum(report.class_counts.values()), len(out))
        self.assertEqual(set(out["churn"]), {0, 1})

    def test_too_little_usable_data_is_refused(self):
        df = datagen.generate(400, dirty=False)
        df["contract"] = "Prepaid"
        with self.assertRaises(DataError) as ctx:
            P.clean(df)
        self.assertIn("usable rows remain", str(ctx.exception))
        df = datagen.generate(400, dirty=False)
        df["churn"] = "No"
        with self.assertRaises(DataError) as ctx:
            P.clean(df)
        self.assertIn("who churned remain", str(ctx.exception))


class TestFeatures(TempDirTest):
    def test_engineered_features_never_use_the_label(self):
        df = datagen.generate(50, dirty=False).drop(columns=[S.ID, S.TARGET])     # no label column at all
        out = P.add_features(df)
        self.assertTrue(set(S.ENGINEERED) <= set(out.columns))
        self.assertFalse(any(S.TARGET in name for name in out.columns))

    def test_engineered_values(self):
        df = pd.DataFrame([{"tenure_months": 10, "monthly_charges": 80.0, "total_charges": 700.0, "tech_support": 1,
                            "online_security": 1}, {"tenure_months": 0, "monthly_charges": 50.0, "total_charges": 0.0,
                                                    "tech_support": 0, "online_security": 1}])
        out = P.add_features(df)
        self.assertAlmostEqual(out["avg_monthly_spend"].iloc[0], 70.0)
        self.assertAlmostEqual(out["price_change"].iloc[0], 10.0)                  # the bill rose from 70 to 80
        self.assertEqual(list(out["protection_services"]), [2, 1])
        self.assertTrue(np.isfinite(out["avg_monthly_spend"].iloc[1]))             # tenure 0 does not divide by zero

    def test_pipeline_copes_with_gaps_and_unseen_categories(self):
        X, y = P.split_xy(P.clean(datagen.generate(800, dirty=False))[0])
        pipe = M.build_pipeline(M.candidates(fast=True)["Logistic Regression"]).fit(X, y)
        odd = X.head(3).copy()
        odd.loc[odd.index[0], ["tenure_months", "support_calls", "tech_support"]] = np.nan
        odd.loc[odd.index[1], "payment_method"] = "Cheque by post"
        odd.loc[odd.index[2], "internet_service"] = np.nan
        probabilities = pipe.predict_proba(odd)[:, 1]
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
