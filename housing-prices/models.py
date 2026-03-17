from xgboost import XGBRegressor

MODEL_NAME = "xgboost_v1"


def get_model(random_state: int = 42):
    return XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )


def get_model_metadata():
    return {
        "model_name": MODEL_NAME,
        "model_type": "XGBRegressor",
        "notes": "Gradient boosting model for improved non-linear relationships",
    }
