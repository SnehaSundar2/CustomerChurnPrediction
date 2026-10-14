"""Shared fixtures. Training takes a few seconds, so ONE small model is trained for the whole test run."""

import atexit
import shutil
import tempfile
import unittest
from pathlib import Path

from churnpredict import datagen, persistence, training

_SHARED = {}

# a new, month-to-month fiber customer who has called support four times and pays manually: very likely to leave
AT_RISK = {"contract": "Month-to-month", "internet_service": "Fiber", "tenure_months": 5, "monthly_charges": 95,
           "support_calls": 4, "late_payments": 1, "payment_method": "Manual payment", "paperless_billing": "yes",
           "tech_support": "no", "online_security": "no"}
# a long-standing two-year DSL customer with both add-ons and automatic payment: very unlikely to leave
LOYAL = {"contract": "Two year", "internet_service": "DSL", "tenure_months": 40, "monthly_charges": 52,
         "support_calls": 0, "late_payments": 0, "payment_method": "Bank transfer (auto)", "paperless_billing": "no",
         "tech_support": "yes", "online_security": "yes"}


def shared():
    """(TrainingResult, saved model directory, raw training DataFrame) — trained once, in fast mode."""
    if not _SHARED:
        folder = Path(tempfile.mkdtemp(prefix="churnpredict_tests_"))
        atexit.register(shutil.rmtree, folder, ignore_errors=True)
        raw = datagen.generate(2000)
        result = training.train(raw, fast=True, tune=False, cv=3, source="synthetic test data")
        persistence.save(result, folder / "model")
        _SHARED.update(result=result, model_dir=folder / "model", raw=raw, folder=folder)
    return _SHARED["result"], _SHARED["model_dir"], _SHARED["raw"]


class TempDirTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def copy_model(self):
        """A private copy of the trained model that a test may damage (each call makes a fresh copy)."""
        _, model_dir, _ = shared()
        self._copies = getattr(self, "_copies", 0) + 1
        target = self.dir / f"model_copy_{self._copies}"
        shutil.copytree(model_dir, target)
        return target
