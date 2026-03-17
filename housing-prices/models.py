from xgboost import XGBRegressor

MODEL_NAME = "xgboost_mae_tuned"


def get_model(random_state: int = 42):
    return XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:absoluteerror",
        random_state=random_state,
        n_jobs=-1,
    )


def get_model_metadata():
    return {
        "model_name": MODEL_NAME,
        "model_type": "XGBRegressor",
        "notes": "XGBoost MAE tuned: deeper trees (6), lower LR (0.03), more estimators (800)",
    }
