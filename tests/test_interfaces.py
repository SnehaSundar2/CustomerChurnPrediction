"""The web interface (over real HTTP on a random local port) and the interactive menu."""

import http.client
import io
import json
import threading
import urllib.parse
from contextlib import redirect_stdout
from unittest.mock import patch

from churnpredict import datagen, web
from churnpredict.cli import ChurnCLI
from churnpredict.predictor import Predictor
from tests.helpers import AT_RISK, LOYAL, TempDirTest, shared

FORM = {k: str(v) for k, v in AT_RISK.items()}
LOYAL_FORM = {k: str(v) for k, v in LOYAL.items()}


class TestWeb(TempDirTest):
    @classmethod
    def setUpClass(cls):
        cls.server = web.make_server(Predictor.from_directory(shared()[1]), port=0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        conn.close()
        return resp.status, data, {k.lower(): v for k, v in resp.getheaders()}

    def post_form(self, values):
        return self.request("POST", "/", urllib.parse.urlencode(values),
                            {"Content-Type": "application/x-www-form-urlencoded"})

    def post_json(self, payload):
        return self.request("POST", "/api/predict", json.dumps(payload), {"Content-Type": "application/json"})

    def test_pages_and_security_headers(self):
        status, body, headers = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Predict churn", body)
        self.assertNotIn("<script", body.lower())                          # the page needs no scripts at all
        self.assertIn("default-src 'none'", headers["content-security-policy"])
        self.assertEqual(headers["x-content-type-options"], "nosniff")
        self.assertEqual(self.request("GET", "/nope")[0], 404)
        self.assertEqual(self.request("POST", "/elsewhere", b"")[0], 404)

    def test_high_risk_customer_page(self):
        status, body, _ = self.post_form(FORM)
        self.assertEqual(status, 200)
        for part in ("chance of leaving", "High risk", "Make a retention offer", "What could keep this customer",
                     "What drives this risk", "Similar customers", "contract"):
            self.assertIn(part, body)
        self.assertIn("chip risk", body)

    def test_low_risk_customer_page(self):
        status, body, _ = self.post_form(LOYAL_FORM)
        self.assertEqual(status, 200)
        self.assertIn("Low risk", body)
        self.assertIn("No offer needed", body)
        self.assertIn("chip ok", body)
        self.assertNotIn("Make a retention offer", body)

    def test_a_blank_total_uses_tenure_times_monthly(self):
        status, body, _ = self.post_form({**FORM, "total_charges": ""})
        self.assertEqual(status, 200)
        self.assertIn("total charges = 475", body)

    def test_form_errors_keep_the_values_and_name_the_field(self):
        status, body, _ = self.post_form({**FORM, "tenure_months": "99", "contract": "Prepaid"})
        self.assertEqual(status, 422)
        self.assertIn("Tenure must be between 0", body)
        self.assertIn("is not supported", body)
        self.assertIn('value="99"', body)                                  # what the user typed is still there
        self.assertNotIn("chance of leaving", body)

    def test_impossible_combinations_are_refused(self):
        status, body, _ = self.post_form({**FORM, "internet_service": "None", "tech_support": "yes"})
        self.assertEqual(status, 422)
        self.assertIn("needs an internet service", body)

    def test_scripts_in_input_are_escaped(self):
        evil = '"><script>alert(1)</script>'
        status, body, _ = self.post_form({**FORM, "contract": evil, "payment_method": evil, "tenure_months": evil})
        self.assertEqual(status, 422)
        self.assertNotIn("<script>alert", body)
        self.assertIn("&lt;script&gt;", body)

    def test_json_api(self):
        status, body, headers = self.post_json({**AT_RISK})
        data = json.loads(body)
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers["content-type"])
        self.assertEqual(data["risk_tier"], "High")
        self.assertTrue(data["make_offer"])
        self.assertTrue(data["retention_actions"])
        status, body, _ = self.post_json({"contract": "Prepaid", "tenure_months": 500})
        self.assertEqual(status, 422)
        self.assertIn("contract", json.loads(body)["fields"])
        health = json.loads(self.request("GET", "/api/health")[1])
        self.assertEqual(health["status"], "ok")
        self.assertIn("contact_threshold", health)

    def test_bad_requests_are_refused(self):
        self.assertEqual(self.request("POST", "/api/predict", "{}", {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("POST", "/api/predict", "{not json", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/predict", "x" * 30_000, {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/", "x" * 30_000, {"Content-Type": "application/x-www-form-urlencoded"})[0], 413)
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example.com"})[0], 403)   # DNS-rebinding guard
        self.assertEqual(self.request("GET", "/", headers={"Host": f"localhost:{self.port}"})[0], 200)


def run_cli(cli, inputs):
    out = io.StringIO()
    with patch("builtins.input", side_effect=inputs), redirect_stdout(out):
        cli.run()
    return out.getvalue()


# answers for "Predict one customer": contract, internet, tenure, monthly, total (blank = derived), support calls,
# late payments, payment method, paperless, tech support, online security
ANSWERS = ["Month-to-month", "Fiber", "5", "95", "", "4", "1", "Manual payment", "yes", "no", "no"]


class TestCLI(TempDirTest):
    def make(self):
        return ChurnCLI(shared()[1], self.dir / "reports")

    def test_predict_session(self):
        out = run_cli(self.make(), ["1", *ANSWERS, "0"])
        for part in ("Churn chance:", "High risk", "WORTH MAKING", "What could keep this customer", "What drives this risk",
                     "Similar customers"):
            self.assertIn(part, out)

    def test_blank_optional_answers_use_defaults(self):
        out = run_cli(self.make(), ["1", "Two year", "DSL", "30", "50", "", "", "", "", "", "", "", "0"])
        self.assertIn("Assumed for missing details", out)
        self.assertIn("Churn chance:", out)

    def test_invalid_answers_are_asked_again(self):
        answers = ["Prepaid", "Month-to-month", "Fiber", "500", "5", "95", *ANSWERS[4:]]   # bad contract, then bad tenure
        out = run_cli(self.make(), ["1", *answers, "0"])
        self.assertIn("is not supported", out)
        self.assertIn("Tenure must be between 0", out)
        self.assertIn("Churn chance:", out)

    def test_three_bad_answers_cancel_the_prediction(self):
        out = run_cli(self.make(), ["1", "Prepaid", "Monthly plan??", "???", "0"])
        self.assertIn("Too many invalid answers", out)
        self.assertNotIn("Churn chance:", out)

    def test_required_answers_cannot_be_skipped(self):
        out = run_cli(self.make(), ["1", "Month-to-month", "", "Fiber", *ANSWERS[2:], "0"])
        self.assertIn("Internet service is required", out)
        self.assertIn("Churn chance:", out)

    def test_addons_without_internet_are_refused_then_corrected(self):
        answers = ["Month-to-month", "None", "5", "20", "", "1", "0", "UPI", "no", "yes", "no", "no"]   # yes for tech support, then no
        out = run_cli(self.make(), ["1", *answers, "0"])
        self.assertIn("needs an internet service", out)
        self.assertIn("Churn chance:", out)

    def test_what_if_session(self):
        out = run_cli(self.make(), ["2", *ANSWERS, "contract", "Month-to-month, Two year", "0"])
        self.assertIn("different contract", out)
        self.assertIn("Month-to-month", out)
        self.assertIn("Two year", out)

    def test_reports_and_menu_errors(self):
        out = run_cli(self.make(), ["6", "7", "8", "", "99", "abc", "0"])
        self.assertIn("MODEL REPORT", out)
        self.assertIn("MODEL COMPARISON", out)
        self.assertIn("Report saved to", out)
        self.assertTrue((self.dir / "reports" / "model_report.html").exists())
        self.assertEqual(out.count("Invalid option"), 2)

    def test_batch_scoring_through_the_menu(self):
        source = self.dir / "in.csv"
        source.write_text("customer_id,contract,internet_service,tenure_months,monthly_charges\n"
                          "A,Month-to-month,Fiber,5,95\nB,Prepaid,DSL,12,50\nC,Two year,DSL,40,52\n", encoding="utf-8")
        target = self.dir / "out.csv"
        out = run_cli(self.make(), ["3", str(source), str(target), "0"])
        self.assertIn("Scored 2 customer(s)", out)
        self.assertIn("High risk", out)
        self.assertIn("Line 3:", out)
        self.assertTrue(target.exists())
        missing = run_cli(self.make(), ["3", str(self.dir / "nothing.csv"), "", "0"])
        self.assertIn("File not found", missing)

    def test_dataset_tools(self):
        path = self.dir / "mine.csv"
        out = run_cli(self.make(), ["9", "400", "", str(path), "10", str(path), "0"])
        self.assertIn("Saved 400 customers", out)
        self.assertIn("rows read", out)
        self.assertIn("Missing values before cleaning", out)
        out = run_cli(self.make(), ["9", "400", "", str(path), "0"])
        self.assertIn("already exists", out)

    def test_no_model_gives_instructions_and_training_works(self):
        empty = ChurnCLI(self.dir / "nomodel", self.dir / "reports")
        out = run_cli(empty, ["n", "1", "0"])
        self.assertIn("No trained model yet", out)
        self.assertIn("Train one first", out)
        data = self.dir / "train.csv"
        datagen.generate(700).to_csv(data, index=False)
        out = run_cli(empty, ["n", "5", str(data), "", "", "", "y", "0"])          # default campaign settings, quick mode
        self.assertIn("MODEL REPORT", out)
        self.assertIn("Model saved to", out)
        self.assertTrue((self.dir / "nomodel" / "model.joblib").exists())

    def test_bad_campaign_settings_are_refused(self):
        data = self.dir / "train.csv"
        datagen.generate(700).to_csv(data, index=False)
        out = run_cli(ChurnCLI(self.dir / "nomodel2", self.dir / "reports"), ["n", "5", str(data), "-5", "0"])
        self.assertIn("Customer value must be between", out)
