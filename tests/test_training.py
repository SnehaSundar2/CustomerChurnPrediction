"""Model training, comparison, calibration, the campaign threshold and evaluation."""

import copy

import numpy as np
import pandas as pd

from churnpredict import datagen, evaluation as E, models as M, preprocessing as P, report, schema as S, training
from churnpredict.exceptions import DataError, ValidationError
from tests.helpers import TempDirTest, shared


class TestEvaluationMaths(TempDirTest):
    def test_metrics_on_a_tiny_example(self):
        m = E.classification_metrics([1, 1, 0, 0, 0, 0], [0.9, 0.3, 0.6, 0.1, 0.2, 0.05], 0.5)
        self.assertEqual((m["tp"], m["fn"], m["fp"], m["tn"], m["flagged"]), (1, 1, 1, 3, 2))
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (50.0, 50.0, 50.0))
        self.assertAlmostEqual(m["accuracy"], 400 / 6)
        self.assertEqual(E.classification_metrics([1, 1], [0.1, 0.2], 0.5)["precision"], 0.0)   # nobody flagged: defined
        self.assertEqual(E.classification_metrics([0, 0], [0.1, 0.2], 0.5)["recall"], 0.0)      # nobody churned: defined

    def test_ranking_metrics(self):
        perfect = E.ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual((perfect["roc_auc"], perfect["average_precision"]), (1.0, 1.0))
        self.assertEqual(E.ranking_metrics([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])["roc_auc"], 0.0)
        one_class = E.ranking_metrics([1, 1, 1], [0.2, 0.5, 0.9])
        self.assertEqual((one_class["roc_auc"], one_class["average_precision"]), (0.0, 0.0))
        self.assertAlmostEqual(E.ranking_metrics([0, 1], [0.0, 1.0])["brier"], 0.0)
        self.assertAlmostEqual(E.ranking_metrics([0, 1], [1.0, 0.0])["brier"], 1.0)

    def test_campaign_profit_by_hand(self):
        actual, p = [1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1]
        r = E.campaign_profit(actual, p, 0.5, value=1000, cost=100, save_rate=0.5)
        self.assertEqual((r["contacted"], r["churners_reached"], r["saved"]), (2, 1, 0.5))
        self.assertAlmostEqual(r["profit"], 1 * 0.5 * 1000 - 2 * 100)               # 300
        self.assertAlmostEqual(r["profit_per_1000"], 300 / 4 * 1000)
        self.assertEqual(E.campaign_profit(actual, p, 2.0, 1000, 100, 0.5)["profit"], 0.0)   # nobody contacted
        self.assertAlmostEqual(E.break_even_probability(12000, 600, 0.3), 1 / 6)

    def test_threshold_follows_the_economics(self):
        # 30 customers at 80% risk who all churn, 70 customers at 10% risk of whom 14 (20%) churn
        actual = [1] * 30 + [1] * 14 + [0] * 56
        p = [0.8] * 30 + [0.1] * 70
        t = E.choose_contact_threshold(actual, p, value=1000, cost=120, save_rate=0.5)
        self.assertAlmostEqual(t["contact"], 0.24)                                 # cost ÷ (save rate × value)
        self.assertTrue(t["profitable"])
        self.assertAlmostEqual(t["training_profit_per_1000"], (30 * 500 - 30 * 120) / 100 * 1000)   # only the 80% group is contacted
        self.assertEqual(t["empirical_best"], 0.8)                                 # the data-driven choice agrees here
        cheap = E.choose_contact_threshold(actual, p, value=1000, cost=1, save_rate=0.5)
        self.assertAlmostEqual(cheap["contact"], 0.002)                            # nearly free offers: contact almost everybody
        self.assertEqual(E.campaign_profit(actual, p, cheap["contact"], 1000, 1, 0.5)["contacted"], 100)
        dear = E.choose_contact_threshold(actual, p, value=1000, cost=5000, save_rate=0.5)
        self.assertFalse(dear["profitable"])
        self.assertGreater(dear["contact"], 1)                                     # no campaign pays: contact nobody

    def test_with_honest_probabilities_the_best_threshold_is_the_break_even_point(self):
        rng = np.random.default_rng(3)
        p = rng.random(40000)
        actual = (rng.random(40000) < p).astype(int)                               # perfectly calibrated scores
        t = E.choose_contact_threshold(actual, p, value=12000, cost=600, save_rate=0.3)
        self.assertAlmostEqual(t["contact"], 1 / 6)
        self.assertAlmostEqual(t["empirical_best"], t["contact"], delta=0.04)      # searching the data finds the same spot
        at_rule = E.campaign_profit(actual, p, t["contact"])["profit"]
        at_best = E.campaign_profit(actual, p, t["empirical_best"])["profit"]
        self.assertLess(abs(at_best - at_rule) / at_best, 0.01)                    # and gains under 1% by doing so

    def test_policy_table(self):
        rng = np.random.default_rng(4)
        p = rng.random(2000)
        actual = (rng.random(2000) < p).astype(int)
        rows = E.policy_table(actual, p, 0.2, 12000, 600, 0.3)
        self.assertEqual([r["policy"] for r in rows], ["Contact nobody", "Contact everybody", "Use the model",
                                                       "Perfect foresight (impossible)"])
        nobody, everybody, model, perfect = rows
        self.assertEqual((nobody["contacted"], nobody["profit"]), (0, 0.0))
        self.assertEqual(everybody["contacted"], 2000)
        self.assertGreater(model["profit"], everybody["profit"])                   # a sensible list beats contacting all
        self.assertGreaterEqual(perfect["profit"], model["profit"])
        self.assertEqual(perfect["contacted"], int(actual.sum()))

    def test_lift_gains_and_calibration_tables(self):
        actual = np.array([1] * 100 + [0] * 900)
        perfect_scores = np.concatenate([np.full(100, 0.9), np.full(900, 0.05)])
        lift = E.lift_table(actual, perfect_scores)
        self.assertEqual(len(lift), 10)
        self.assertEqual((lift[0]["churn_rate"], lift[0]["lift"], lift[0]["captured"]), (100.0, 10.0, 100.0))
        self.assertEqual(sum(r["churned"] for r in lift), 100)
        self.assertAlmostEqual(lift[-1]["captured"], 100.0)
        gains = E.gains_points(actual, perfect_scores)
        self.assertEqual((gains[0], gains[-1]), ((0.0, 0.0), (1.0, 1.0)))
        self.assertTrue(all(b >= a for (_, a), (_, b) in zip(gains, gains[1:])))   # never goes down
        rng = np.random.default_rng(5)
        p = rng.random(30000)
        cal = E.calibration_table((rng.random(30000) < p).astype(int), p)
        self.assertEqual(len(cal), 10)
        self.assertTrue(all(abs(r["predicted"] - r["observed"]) < 4 for r in cal))
        self.assertEqual(len(E.calibration_table([0, 1], [0.05, 0.95])), 2)       # empty bands are skipped

    def test_segments(self):
        frame = pd.DataFrame({"g": ["a", "a", "b", "b"], "actual": [1, 1, 0, 1], "probability": [0.5, 0.7, 0.2, 0.4]})
        rows = E.segment_rows(frame, "g")
        self.assertEqual([r["group"] for r in rows], ["a", "b"])                   # highest real churn first
        self.assertAlmostEqual(rows[0]["gap"], 60 - 100)
        self.assertAlmostEqual(rows[1]["gap"], 30 - 50)


class TestTrainedModel(TempDirTest):
    def test_comparison_table(self):
        result, _, _ = shared()
        names = [c["model"] for c in result.comparison]
        self.assertEqual(set(names), {M.BASELINE, "Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"})
        aucs = [c["roc_auc"] for c in result.comparison]
        self.assertEqual(aucs, sorted(aucs, reverse=True))                         # best first
        base = result.comparison[-1]
        self.assertEqual(base["model"], M.BASELINE)                                # the baseline is the worst
        self.assertAlmostEqual(base["roc_auc"], 0.5, places=2)                     # it ranks nobody
        self.assertGreater(result.comparison[0]["roc_auc"], 0.78)
        self.assertGreater(base["brier"], result.comparison[0]["brier"])
        self.assertNotEqual(result.metadata["model"], M.BASELINE)

    def test_split_sizes_and_churn_share(self):
        result, _, raw = shared()
        rows = result.metadata["rows"]
        self.assertEqual(rows["raw"], len(raw))
        self.assertEqual(rows["train"] + rows["test"], rows["clean"])
        self.assertAlmostEqual(rows["test"] / rows["clean"], 0.2, delta=0.02)
        self.assertEqual(len(result.test_frame), rows["test"])
        self.assertEqual(len(result.reference), rows["train"])
        self.assertAlmostEqual(result.reference["churn"].mean(), result.test_frame["actual"].mean(), delta=0.02)

    def test_accuracy_on_unseen_customers(self):
        result, _, _ = shared()
        test, train = result.metadata["metrics"]["test"], result.metadata["metrics"]["train"]
        base = result.metadata["metrics"]["baseline_cv"]
        self.assertGreater(test["roc_auc"], 0.78)
        self.assertGreater(test["average_precision"], test["churn_rate"] / 100 + 0.2)   # far above "guess the average"
        self.assertLess(test["brier"], base["brier"])                              # more honest than the average guess
        self.assertGreater(test["recall"], 60)
        self.assertGreater(test["precision"], 1.4 * test["churn_rate"])            # flagged customers are far likelier to leave
        self.assertLess(train["roc_auc"] - test["roc_auc"], 0.09)                  # no serious overfitting
        self.assertEqual(test["tp"] + test["fp"] + test["fn"] + test["tn"], test["n"])
        self.assertEqual(test["tp"] + test["fp"], test["flagged"])

    def test_the_default_cut_off_would_miss_many_churners(self):
        result, _, _ = shared()
        test = result.metadata["metrics"]["test"]
        self.assertLess(test["default_threshold"]["recall"], test["recall"])       # 0.5 is the wrong cut-off for this business
        self.assertLess(test["default_threshold"]["flagged"], test["flagged"])

    def test_threshold_and_campaign_economics(self):
        result, _, _ = shared()
        meta = result.metadata
        eco, t = meta["economics"], meta["thresholds"]
        self.assertTrue(eco["profitable"])
        self.assertAlmostEqual(eco["break_even"], 600 / (0.3 * 12000))
        self.assertAlmostEqual(t["contact"], eco["break_even"])                    # offer when expected saving exceeds the cost
        self.assertEqual(t["watch"], t["contact"] / 2)
        self.assertGreater(eco["empirical_best_profit_per_1000"], 0)
        policies = {p["policy"]: p for p in meta["policies"]}
        model, everybody = policies["Use the model"], policies["Contact everybody"]
        self.assertGreater(model["profit"], 0)
        self.assertGreater(model["profit"], everybody["profit"])                   # the model's list beats offering to all
        self.assertLessEqual(model["profit"], policies["Perfect foresight (impossible)"]["profit"])
        frame = result.test_frame
        self.assertTrue(((frame["probability"] >= t["contact"]).astype(int) == frame["predicted"]).all())

    def test_calibration_decision_is_recorded(self):
        result, _, _ = shared()
        cal = result.metadata["calibration"]
        self.assertIn(cal["method"], {"none", "sigmoid", "isotonic"})
        self.assertEqual(cal["used"], cal["method"] != "none")
        self.assertLessEqual(min(cal["brier_before"], cal["brier_after"]), max(cal["brier_before"], cal["brier_after"]))
        probabilities = result.test_frame["probability"]
        self.assertTrue(probabilities.between(0, 1).all())
        cal_rows = result.metadata["calibration_table"]
        self.assertTrue(all(abs(r["predicted"] - r["observed"]) < 25 for r in cal_rows if r["customers"] >= 30))

    def test_importance_and_ceiling(self):
        result, _, _ = shared()
        imp = result.importance
        self.assertEqual([i["importance"] for i in imp], sorted((i["importance"] for i in imp), reverse=True))
        self.assertTrue(all(i["importance"] >= 0 for i in imp))
        self.assertIn("contract", {i["feature"] for i in imp[:3]})
        self.assertEqual({i["feature"] for i in imp}, set(S.RAW_FEATURES))
        ceiling = result.metadata["ceiling"]
        self.assertIsNotNone(ceiling)
        self.assertLess(result.metadata["metrics"]["test"]["roc_auc"], ceiling["roc_auc"] + 0.05)

    def test_tables_and_curves_in_metadata(self):
        result, _, _ = shared()
        meta = result.metadata
        self.assertEqual(len(meta["lift"]), 10)
        self.assertGreater(meta["lift"][0]["churn_rate"], meta["lift"][-1]["churn_rate"] + 20)
        self.assertEqual(sum(r["customers"] for r in meta["lift"]), meta["rows"]["test"])
        self.assertEqual(len(meta["profit_curve"]), 51)
        best_on_curve = max(v for _, v in meta["profit_curve"])
        self.assertGreater(best_on_curve, 0)
        self.assertEqual(set(meta["segments"]), {"contract", "internet", "payment"})
        month = next(s for s in meta["segments"]["contract"] if s["group"] == "Month-to-month")
        two = next(s for s in meta["segments"]["contract"] if s["group"] == "Two year")
        self.assertGreater(month["actual"], two["actual"] * 2)                     # the segments differ as the rule says

    def test_metadata_is_complete(self):
        result, _, _ = shared()
        for key in ("trained_at", "model", "metrics", "economics", "thresholds", "policies", "calibration", "environment",
                    "training_ranges", "typical", "importance", "comparison", "cleaning", "segments", "lift", "gains",
                    "calibration_table", "profit_curve", "seed", "class_counts"):
            self.assertIn(key, result.metadata, key)
        self.assertEqual(set(result.metadata["typical"]), set(S.RAW_FEATURES))


class TestReproducibilityAndGuards(TempDirTest):
    def test_same_seed_same_result(self):
        raw = datagen.generate(700, seed=11)
        a = training.train(raw, fast=True, tune=False, cv=3)
        b = training.train(raw, fast=True, tune=False, cv=3)
        self.assertEqual(a.metadata["thresholds"], b.metadata["thresholds"])
        np.testing.assert_allclose(a.test_frame["probability"], b.test_frame["probability"])
        self.assertEqual(list(a.test_frame.index), list(b.test_frame.index))

    def test_campaign_settings_reach_the_model(self):
        raw = datagen.generate(700, seed=12)
        result = training.train(raw, fast=True, tune=False, cv=3, value=15000, cost=1500, save_rate=0.25)
        eco = result.metadata["economics"]
        self.assertEqual((eco["value"], eco["cost"], eco["save_rate"]), (15000.0, 1500.0, 0.25))
        self.assertAlmostEqual(eco["break_even"], 0.4)
        self.assertAlmostEqual(result.metadata["thresholds"]["contact"], 0.4)      # costlier offers: be pickier

    def test_bad_settings_and_unusable_data_are_refused(self):
        raw = datagen.generate(700, seed=13)
        for kwargs in ({"cost": -5}, {"value": 0}, {"save_rate": 1.5}, {"save_rate": 0}, {"cost": float("nan")}):
            with self.assertRaises(ValidationError, msg=str(kwargs)):
                training.train(raw, fast=True, tune=False, **kwargs)
        with self.assertRaises(DataError):
            training.train(datagen.generate(100, dirty=False), fast=True, tune=False)
        stayed = datagen.generate(500, dirty=False)
        stayed["churn"] = "No"
        with self.assertRaises(DataError):
            training.train(stayed, fast=True, tune=False)

    def test_tuning_and_calibration_helpers(self):
        X, y = P.split_xy(P.clean(datagen.generate(800, dirty=False))[0])
        search, params = M.tune("Decision Tree", X, y, cv=3, iterations=3)
        self.assertIn(params["max_depth"], M.TUNING_GRIDS["Decision Tree"]["model__max_depth"])
        self.assertEqual(M.tune(M.BASELINE, X, y), (None, {}))                     # nothing to tune
        pipe = M.pipeline_with_params("Decision Tree", params)
        self.assertEqual(pipe.fit(X, y).predict_proba(X.head(3)).shape, (3, 2))
        model, oof, info = M.choose_calibration(M.build_pipeline(M.candidates(fast=True)["Random Forest"]), X, y, cv=3)
        self.assertEqual(oof.shape, (len(y),))
        self.assertTrue(((oof >= 0) & (oof <= 1)).all())
        self.assertIn(info["method"], {"none", "sigmoid", "isotonic"})
        self.assertEqual(model.fit(X, y).predict_proba(X.head(2)).shape, (2, 2))   # the chosen model is ready to train


class TestReports(TempDirTest):
    def test_text_and_html_reports(self):
        meta = shared()[0].metadata
        text = report.model_text(meta)
        for part in ("ACCURACY ON THE UNSEEN TEST SET", "CONFUSION MATRIX", "WHAT A RETENTION CAMPAIGN EARNS",
                     "LIFT BY RISK DECILE", "WHAT THE MODEL RELIES ON", "CHURN BY CONTRACT", "ROC AUC"):
            self.assertIn(part, text)
        self.assertIn("₹", text)
        self.assertIn("chosen", report.comparison_text(meta))
        page = report.render_html(meta)
        self.assertEqual(page.count('<details class="tv">'), 6)                    # every chart has a table view
        self.assertIn("prefers-color-scheme:dark", page)
        self.assertIn("Data cleaning log", page)
        self.assertIn("What a retention campaign earns", page)
        path = report.save_html(meta, self.dir / "out" / "report.html")
        self.assertTrue(path.exists())

    def test_text_in_the_report_is_escaped(self):
        meta = copy.deepcopy(shared()[0].metadata)
        meta["segments"]["contract"][0]["group"] = "<script>alert(1)</script>"
        page = report.render_html(meta)
        self.assertNotIn("<script>alert", page)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", page)

    def test_chart_helpers(self):
        for value in (7, 45, 130, 999):
            top = report._tick_ceiling(value, 5)
            self.assertGreaterEqual(top, value)
        self.assertEqual(report._nice_axis(0, 1), (0, 1.0, 0.2))
        low, high, step = report._nice_axis(-120, 540)
        self.assertTrue(low <= -120 and high >= 540 and step > 0)
        svg = report.xy_chart([(0.0, -100), (0.5, 400), (1.0, 100)], "t", "x", "y", y_fmt="{:,.0f}", marker=(0.5, 400),
                              marker_label="best")
        self.assertIn('class="line"', svg)
        self.assertIn("best", svg)
        self.assertIn('class="baseline"', svg)                                     # the zero line is drawn when profits go negative
        dots = report.xy_chart([(0.1, 0.12, "a"), (0.5, 0.45, "b")], "t", "x", "y", dots=True, diagonal=True, y_range=(0, 1, 0.2))
        self.assertEqual(dots.count("<circle"), 2)
        self.assertIn('class="ref"', dots)
        columns = report.column_chart([("1", 85), ("2", 53), ("3", 0)], "t")
        self.assertEqual(columns.count("<path"), 2)                                # zero-height columns are not drawn
