from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

MODEL_NAME = "ensemble_xgb_lgbm_catboost"


def get_models(random_state: int = 42):
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
        early_stopping_rounds=20,
    )
    lgbm = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="mse",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )
    catboost = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=8,
        min_data_in_leaf=5,
        subsample=0.8,
        loss_function="RMSE",
        random_state=random_state,
        verbose=0,
        early_stopping_rounds=20,
    )
    return [
        ("xgb", xgb),
        ("lgbm", lgbm),
        ("catboost", catboost),
    ]


def get_model(random_state: int = 42):
    return get_models(random_state)


def get_model_metadata():
    return {
        "model_name": MODEL_NAME,
        "model_type": "Ensemble(XGBRegressor + LGBMRegressor + CatBoostRegressor)",
        "notes": "Stacked ensemble with Ridge meta-learner combining XGBoost, LightGBM, and CatBoost with RMSE loss",
    }
