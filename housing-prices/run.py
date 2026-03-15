import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from features import build_feature_matrices
from models import get_model, get_model_metadata

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

APPROVED_RUN_PATH = ROOT / "approved_run.json"
RESEARCH_LOG_PATH = ROOT / "research_log.md"

THRESHOLD = -50
RANDOM_STATE = 42

HYPOTHESIS = "Adding ordinal encoding for BsmtExposure (basement walkout/garden exposure) will capture basement-quality price variations and reduce CV RMSE"


def ensure_dirs():
    ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    return train_df, test_df


def evaluate_current_model(x_train, y_train):
    model = get_model(random_state=RANDOM_STATE)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(
        model, x_train, y_train, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1
    )
    rmse_scores = -scores
    return {
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std": float(rmse_scores.std()),
        "fold_scores": [float(x) for x in rmse_scores],
        "n_features": int(x_train.shape[1]),
        "n_rows": int(x_train.shape[0]),
    }


def fit_and_predict(x_train, y_train, x_test):
    model = get_model(random_state=RANDOM_STATE)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return preds


def load_approved_run():
    if APPROVED_RUN_PATH.exists():
        return json.loads(APPROVED_RUN_PATH.read_text(encoding="utf-8"))
    return None


def decide(current, approved):
    if approved is None:
        return {
            "status": "no_baseline",
            "accepted": True,
            "delta_cv_rmse": None,
            "reason": "No approved baseline found; current run becomes the initial baseline candidate.",
        }

    delta = current["cv_rmse_mean"] - approved["metrics"]["cv_rmse_mean"]

    if delta <= THRESHOLD:
        return {
            "status": "accepted",
            "accepted": True,
            "delta_cv_rmse": float(delta),
            "reason": f"Reduced RMSE by {-delta:.2f}, meeting threshold of {-THRESHOLD:.2f} improvement.",
        }

    return {
        "status": "rejected",
        "accepted": False,
        "delta_cv_rmse": float(delta),
        "reason": f"RMSE reduction {-delta:.2f} did not meet threshold of {-THRESHOLD:.2f} improvement.",
    }


def write_submission(test_df, preds):
    submission = pd.DataFrame(
        {
            "Id": test_df["Id"],
            "SalePrice": preds,
        }
    )
    submission.to_csv(ARTIFACTS_DIR / "submission.csv", index=False)


def write_current_run_json(current_run):
    (ARTIFACTS_DIR / "current_run.json").write_text(
        json.dumps(current_run, indent=2),
        encoding="utf-8",
    )


def ensure_research_log():
    if not RESEARCH_LOG_PATH.exists():
        RESEARCH_LOG_PATH.write_text(
            "\n".join(
                [
                    "# Housing Prices autoresearch log",
                    "",
                    "| timestamp_utc | branch | model_name | feature_set | cv_rmse_mean | cv_rmse_std | baseline_cv_rmse | delta_cv_rmse | threshold | decision | hypothesis | notes |",
                    "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
                ]
            ),
            encoding="utf-8",
        )


def append_research_log(current_run):
    ensure_research_log()

    baseline_val = (
        f"{current_run['approved_baseline']['metrics']['cv_rmse_mean']:.2f}"
        if current_run["approved_baseline"] is not None
        else ""
    )
    delta_val = (
        f"{current_run['decision']['delta_cv_rmse']:.2f}"
        if current_run["decision"]["delta_cv_rmse"] is not None
        else ""
    )

    line = (
        f"| {current_run['timestamp_utc']} "
        f"| {current_run['branch']} "
        f"| {current_run['model']['model_name']} "
        f"| {current_run['feature_set']} "
        f"| {current_run['metrics']['cv_rmse_mean']:.2f} "
        f"| {current_run['metrics']['cv_rmse_std']:.2f} "
        f"| {baseline_val} "
        f"| {delta_val} "
        f"| {THRESHOLD} "
        f"| {current_run['decision']['status']} "
        f"| {current_run['hypothesis']} "
        f"| {current_run['decision']['reason']} |"
    )

    with RESEARCH_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def detect_branch_name():
    head_path = ROOT / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"
    head = head_path.read_text(encoding="utf-8").strip()
    if head.startswith("ref:"):
        return head.split("/")[-1]
    return "detached"


def main():
    ensure_dirs()

    train_df, test_df = load_data()
    feature_set = "baseline"

    x_train, y_train, x_test = build_feature_matrices(
        train_df, test_df, feature_set=feature_set
    )

    metrics = evaluate_current_model(x_train, y_train)
    preds = fit_and_predict(x_train, y_train, x_test)
    write_submission(test_df, preds)

    approved = load_approved_run()

    current_run = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "branch": detect_branch_name(),
        "feature_set": feature_set,
        "hypothesis": HYPOTHESIS,
        "model": get_model_metadata(),
        "metrics": metrics,
        "approved_baseline": approved,
    }
    current_run["decision"] = decide(current_run["metrics"], approved)

    write_current_run_json(current_run)
    append_research_log(current_run)

    print(json.dumps(current_run, indent=2))


if __name__ == "__main__":
    main()
