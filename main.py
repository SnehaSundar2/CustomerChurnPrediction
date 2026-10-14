"""Entry point for the Customer Churn Prediction project.

Usage:
    python main.py                                       # interactive menu
    python main.py --train                               # train on generated sample customers
    python main.py --train --data my_customers.csv       # train on your own CSV (see README for columns)
    python main.py --train --offer-cost 800 --customer-value 15000 --save-rate 0.25
    python main.py --predict contract="Month-to-month" internet_service=Fiber tenure_months=5 monthly_charges=95
    python main.py --serve --port 8030                   # local web interface
    python main.py --report reports/model_report.html    # HTML report of the trained model
    python main.py --generate sample_data/customers.csv --rows 3000
"""

import argparse
import logging
import sys
from pathlib import Path

from churnpredict import datagen, persistence, report, schema as S, training
from churnpredict import preprocessing as P
from churnpredict import validators as v
from churnpredict.cli import SYNTHETIC_ROWS, ChurnCLI
from churnpredict.exceptions import ChurnError
from churnpredict.predictor import require_predictor
from churnpredict.web import serve


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):  # show ₹ and block characters on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Customer Churn Prediction (machine learning project)")
    parser.add_argument("--model-dir", default="models", help="folder for the trained model (default models)")
    parser.add_argument("--train", action="store_true", help="train a model and save it")
    parser.add_argument("--data", metavar="CSV", help="with --train: your own dataset (default: generated sample data)")
    parser.add_argument("--rows", default=str(SYNTHETIC_ROWS), help="customers to generate (default 3000)")
    parser.add_argument("--fast", action="store_true", help="with --train: smaller models, much faster")
    parser.add_argument("--no-tune", action="store_true", help="with --train: skip hyper-parameter tuning")
    parser.add_argument("--customer-value", default=str(S.CUSTOMER_VALUE), help="with --train: rupees kept when a would-be churner is saved")
    parser.add_argument("--offer-cost", default=str(S.OFFER_COST), help="with --train: rupees to make one retention offer")
    parser.add_argument("--save-rate", default=str(S.SAVE_RATE), help="with --train: share of would-be churners an offer saves (0-1)")
    parser.add_argument("--predict", nargs="+", metavar="KEY=VALUE", help="predict one customer, e.g. contract=Two year tenure_months=30 ...")
    parser.add_argument("--serve", action="store_true", help="start the local web interface")
    parser.add_argument("--port", default="8030", help="web interface port (default 8030)")
    parser.add_argument("--report", metavar="HTML", help="save an HTML report of the trained model")
    parser.add_argument("--generate", metavar="CSV", help="write a synthetic dataset to a CSV file")
    parser.add_argument("--clean-data", action="store_true", help="with --generate: a dataset without data problems")
    args = parser.parse_args(argv)

    logging.basicConfig(filename="churnpredict.log", level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cli = ChurnCLI(args.model_dir)
    try:
        if args.generate:
            rows = v.validate_int(args.rows, "Rows", 200, 50000)
            path = Path(args.generate)
            if path.exists():
                print(f"Error: {path} already exists; choose another name.")
                return 1
            path.parent.mkdir(parents=True, exist_ok=True)
            datagen.generate(rows, dirty=not args.clean_data).to_csv(path, index=False)
            print(f"Saved {rows} synthetic customers to {path}.")
            return 0
        if args.train:
            value = v.validate_number(args.customer_value, "Customer value", 1, 10_000_000)
            cost = v.validate_number(args.offer_cost, "Offer cost", 1, 10_000_000)
            save = v.validate_number(args.save_rate, "Save rate", 0.01, 1)
            if args.data:
                df, label = P.load_dataset(args.data), Path(args.data).name
            else:
                df, label = datagen.generate(v.validate_int(args.rows, "Rows", 500, 50000)), "synthetic (generated)"
            result = training.train(df, fast=args.fast, tune=not (args.fast or args.no_tune), source=label,
                                    progress=lambda m: print(f"  {m}"), value=value, cost=cost, save_rate=save)
            persistence.save(result, args.model_dir)
            print("\n" + "\n".join(result.cleaning.lines()) + "\n")
            print(report.comparison_text(result.metadata) + "\n")
            print(report.model_text(result.metadata))
            print(f"\nModel saved to {args.model_dir}/")
            return 0
        if args.predict:
            raw = {}
            for pair in args.predict:
                if "=" not in pair:
                    print(f"Error: '{pair}' should look like key=value, e.g. tenure_months=5.")
                    return 1
                key, value = pair.split("=", 1)
                raw[key] = value
            cli.show_prediction(require_predictor(args.model_dir).predict(raw))
            return 0
        if args.report:
            print(f"Report saved to {report.save_html(require_predictor(args.model_dir).meta, args.report)}")
            return 0
        if args.serve:
            serve(require_predictor(args.model_dir), v.validate_port(args.port))
            return 0
        cli.run()
        return 0
    except ChurnError as exc:
        print(f"Error: {exc}")
        return 1
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
