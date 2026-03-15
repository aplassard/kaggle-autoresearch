from sklearn.linear_model import Ridge

MODEL_NAME = "ridge_v1"


def get_model(random_state: int = 42):
    return Ridge(alpha=10.0, random_state=random_state)


def get_model_metadata():
    return {
        "model_name": MODEL_NAME,
        "model_type": "Ridge",
        "notes": "Single active model for branch-based autoresearch workflow",
    }
