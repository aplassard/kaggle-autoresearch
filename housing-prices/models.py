from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

MODEL_NAME = "ensemble_xgb_lgbm"


def get_models(random_state: int = 42):
    xgb = XGBRegressor(
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
    lgbm = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=6,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="mae",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    return [("xgb", xgb), ("lgbm", lgbm)]


def get_model(random_state: int = 42):
    return get_models(random_state)


def get_model_metadata():
    return {
        "model_name": MODEL_NAME,
        "model_type": "Ensemble(XGBRegressor + LGBMRegressor)",
        "notes": "Simple average ensemble of tuned XGBoost and LightGBM with MAE loss",
    }
