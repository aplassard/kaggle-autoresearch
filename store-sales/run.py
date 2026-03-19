import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

from features import build_feature_matrices
from models import get_model, get_models, get_model_metadata

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

APPROVED_RUN_PATH = ROOT / "approved_run.json"
RESEARCH_LOG_PATH = ROOT / "research_log.md"

THRESHOLD = -0.005
RANDOM_STATE = 42

HYPOTHESIS = "Adding a 14-day sales lag will capture bi-weekly patterns related to paydays (15th and last day of month), complementing the 7-day lag and improving forecasting accuracy"


def ensure_dirs():
    ARTIFACTS_DIR.mkdir(exist_ok=True)


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    test_df = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["date"])
    stores_df = pd.read_csv(DATA_DIR / "stores.csv")
    oil_df = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays_df = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    return train_df, test_df, stores_df, oil_df, holidays_df


def rmsle(y_true, y_pred):
    y_true = np.maximum(y_true, 0)
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))


def evaluate_current_model(x_train, y_train, groups):
    y_train_log = np.log1p(y_train)
    models = get_models(random_state=RANDOM_STATE)
    unique_dates = sorted(groups.unique())
    n_splits = 5

    fold_size = len(unique_dates) // (n_splits + 1)
    fold_scores = []

    for i in range(n_splits):
        cutoff_idx = (i + 1) * fold_size
        cutoff_date = unique_dates[cutoff_idx]

        train_mask = groups < cutoff_date
        val_mask = groups >= cutoff_date

        if i < n_splits - 1:
            next_cutoff = unique_dates[min((i + 2) * fold_size, len(unique_dates) - 1)]
            val_mask = (groups >= cutoff_date) & (groups < next_cutoff)

        x_tr = x_train[train_mask]
        x_val = x_train[val_mask]
        y_tr = y_train_log[train_mask]
        y_val = y_train_log[val_mask]

        if len(x_tr) == 0 or len(x_val) == 0:
            continue

        oof_preds = np.zeros((len(x_val), len(models)))
        for j, (name, model) in enumerate(models):
            if name == "catboost":
                model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)], verbose=False)
            else:
                model.fit(x_tr, y_tr, eval_set=[(x_val, y_val)])
            oof_preds[:, j] = model.predict(x_val)

        meta_learner = Ridge(alpha=1.0)
        meta_learner.fit(oof_preds, y_val)
        stacked_preds_log = meta_learner.predict(oof_preds)
        stacked_preds = np.expm1(stacked_preds_log)

        score = rmsle(y_train[val_mask], stacked_preds)
        fold_scores.append(score)

    return {
        "cv_rmsle_mean": float(np.mean(fold_scores)),
        "cv_rmsle_std": float(np.std(fold_scores)),
        "fold_scores": [float(x) for x in fold_scores],
        "n_features": int(x_train.shape[1]),
        "n_rows": int(x_train.shape[0]),
    }


def fit_and_predict(x_train, y_train, x_test):
    y_train_log = np.log1p(y_train)
    models = get_models(random_state=RANDOM_STATE)

    oof_preds = np.zeros((len(x_train), len(models)))
    n_splits = 5
    dates = np.arange(len(x_train))
    fold_size = len(x_train) // (n_splits + 1)

    for i in range(n_splits):
        cutoff = (i + 1) * fold_size
        train_idx = np.arange(cutoff)
        val_start = cutoff
        val_end = (
            min((i + 2) * fold_size, len(x_train)) if i < n_splits - 1 else len(x_train)
        )
        val_idx = np.arange(val_start, val_end)

        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        x_tr = x_train[train_idx]
        y_tr = y_train_log[train_idx]

        for j, (name, model) in enumerate(models):
            if name == "catboost":
                model.fit(
                    x_tr,
                    y_tr,
                    eval_set=[(x_train[val_idx], y_train_log[val_idx])],
                    verbose=False,
                )
            else:
                model.fit(
                    x_tr, y_tr, eval_set=[(x_train[val_idx], y_train_log[val_idx])]
                )
            oof_preds[val_idx, j] = model.predict(x_train[val_idx])

    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(oof_preds, y_train_log)

    val_size = int(len(x_train) * 0.1)
    x_tr_final = x_train[:-val_size]
    y_tr_final = y_train_log[:-val_size]
    x_val_final = x_train[-val_size:]
    y_val_final = y_train_log[-val_size:]

    test_preds = np.zeros((len(x_test), len(models)))
    for j, (name, model) in enumerate(models):
        if name == "catboost":
            model.fit(
                x_tr_final,
                y_tr_final,
                eval_set=[(x_val_final, y_val_final)],
                verbose=False,
            )
        else:
            model.fit(x_tr_final, y_tr_final, eval_set=[(x_val_final, y_val_final)])
        test_preds[:, j] = model.predict(x_test)

    return np.expm1(meta_learner.predict(test_preds))


def load_approved_run():
    if APPROVED_RUN_PATH.exists():
        return json.loads(APPROVED_RUN_PATH.read_text(encoding="utf-8"))
    return None


def decide(current, approved):
    if approved is None:
        return {
            "status": "no_baseline",
            "accepted": True,
            "delta_cv_rmsle": None,
            "reason": "No approved baseline found; current run becomes the initial baseline candidate.",
        }

    delta = current["cv_rmsle_mean"] - approved["metrics"]["cv_rmsle_mean"]

    if delta <= THRESHOLD:
        return {
            "status": "accepted",
            "accepted": True,
            "delta_cv_rmsle": float(delta),
            "reason": f"Reduced RMSLE by {-delta:.5f}, meeting threshold of {-THRESHOLD:.5f} improvement.",
        }

    return {
        "status": "rejected",
        "accepted": False,
        "delta_cv_rmsle": float(delta),
        "reason": f"RMSLE reduction {-delta:.5f} did not meet threshold of {-THRESHOLD:.5f} improvement.",
    }


def write_submission(test_df, preds):
    preds = np.maximum(preds, 0)
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "sales": preds,
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
                    "# Store Sales autoresearch log",
                    "",
                    "| timestamp_utc | branch | model_name | feature_set | cv_rmsle_mean | cv_rmsle_std | baseline_cv_rmsle | delta_cv_rmsle | threshold | decision | hypothesis | notes |",
                    "|---|---|---|---|---:|---:|---:|---:|---:|---|---|",
                ]
            ),
            encoding="utf-8",
        )


def append_research_log(current_run):
    ensure_research_log()

    baseline_val = (
        f"{current_run['approved_baseline']['metrics']['cv_rmsle_mean']:.5f}"
        if current_run["approved_baseline"] is not None
        else ""
    )
    delta_val = (
        f"{current_run['decision']['delta_cv_rmsle']:.5f}"
        if current_run["decision"]["delta_cv_rmsle"] is not None
        else ""
    )

    line = (
        f"| {current_run['timestamp_utc']} "
        f"| {current_run['branch']} "
        f"| {current_run['model']['model_name']} "
        f"| {current_run['feature_set']} "
        f"| {current_run['metrics']['cv_rmsle_mean']:.5f} "
        f"| {current_run['metrics']['cv_rmsle_std']:.5f} "
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

    train_df, test_df, stores_df, oil_df, holidays_df = load_data()
    feature_set = "baseline"

    x_train, y_train, x_test, train_dates = build_feature_matrices(
        train_df, test_df, stores_df, oil_df, holidays_df, feature_set=feature_set
    )

    metrics = evaluate_current_model(x_train.values, y_train.values, train_dates)
    preds = fit_and_predict(x_train.values, y_train.values, x_test.values)
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
