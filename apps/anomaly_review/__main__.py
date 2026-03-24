"""Entry point for `python -m apps.anomaly_review`."""

import sys

from apps.anomaly_review.main import run_app

if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else None
    feedback_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_app(data_path, feedback_path)
