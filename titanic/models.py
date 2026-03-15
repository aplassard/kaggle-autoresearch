from sklearn.ensemble import RandomForestClassifier

MODEL_NAME = "random_forest_v1"


def get_model(random_state: int = 42):
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        min_samples_split=6,
        random_state=random_state,
        n_jobs=-1,
    )


def get_model_metadata():
    return {
        "model_name": MODEL_NAME,
        "model_type": "RandomForestClassifier",
        "notes": "Single active model for branch-based autoresearch workflow",
    }
