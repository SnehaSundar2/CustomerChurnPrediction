"""Customer Churn Prediction.

An end-to-end machine-learning project: data generation and cleaning, feature engineering, model comparison
with cross-validation, probability calibration, a contact threshold chosen from the economics of a
retention campaign, honest evaluation on a held-out test set, explanations and retention actions, and
prediction interfaces (command-line menu, local web form, JSON API, batch CSV).
"""

import logging

__version__ = "1.0.0"

logging.getLogger("churnpredict").addHandler(logging.NullHandler())
