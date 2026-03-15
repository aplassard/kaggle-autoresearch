import re
import pandas as pd

BASELINE_COLUMNS = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

ENGINEERED_COLUMNS = [
    "FamilySize",
    "IsAlone",
    "HasCabin",
    "CabinDeck",
    "Title",
    "TicketPrefix",
    "FarePerPerson",
    "IsSenior",
]


def extract_title(name: str) -> str:
    match = re.search(r",\s*([^\.]+)\.", str(name))
    return match.group(1).strip() if match else "Unknown"


def normalize_title(title: str) -> str:
    mapping = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Lady": "Rare",
        "Countess": "Rare",
        "Capt": "Rare",
        "Col": "Rare",
        "Don": "Rare",
        "Dr": "Rare",
        "Major": "Rare",
        "Rev": "Rare",
        "Sir": "Rare",
        "Jonkheer": "Rare",
        "Dona": "Rare",
    }
    title = str(title)
    return mapping.get(
        title, title if title in {"Mr", "Miss", "Mrs", "Master"} else "Rare"
    )


def extract_cabin_deck(cabin: str) -> str:
    if pd.isna(cabin) or str(cabin).strip() == "":
        return "Unknown"
    return str(cabin).strip()[0]


def extract_ticket_prefix(ticket: str) -> str:
    if pd.isna(ticket):
        return "NONE"
    t = str(ticket).upper().replace(".", "").replace("/", " ").strip()
    parts = t.split()
    alpha = [p for p in parts if not p.isdigit()]
    return "_".join(alpha) if alpha else "NONE"


def fill_age(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(["Pclass", "Sex"])["Age"].transform("median")
    out["Age"] = out["Age"].fillna(grouped)
    out["Age"] = out["Age"].fillna(out["Age"].median())
    return out


def make_features(df: pd.DataFrame, feature_set: str = "engineered") -> pd.DataFrame:
    out = df.copy()

    out["Sex"] = out["Sex"].fillna("Unknown")
    out["Embarked"] = out["Embarked"].fillna(out["Embarked"].mode().iloc[0])
    out["Fare"] = out["Fare"].fillna(out["Fare"].median())
    out = fill_age(out)

    if feature_set == "engineered":
        out["FamilySize"] = out["SibSp"].fillna(0) + out["Parch"].fillna(0) + 1
        out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
        out["HasCabin"] = out["Cabin"].notna().astype(int)
        out["CabinDeck"] = out["Cabin"].map(extract_cabin_deck)
        out["Title"] = out["Name"].map(extract_title).map(normalize_title)
        out["TicketPrefix"] = out["Ticket"].map(extract_ticket_prefix)
        out["FarePerPerson"] = out["Fare"] / out["FamilySize"]
        out["IsSenior"] = (out["Age"] >= 60).astype(int)
        cols = BASELINE_COLUMNS + ENGINEERED_COLUMNS
    else:
        cols = BASELINE_COLUMNS

    return out[cols]


def build_feature_matrices(train_df, test_df, feature_set="engineered"):
    y = train_df["Survived"].astype(int).reset_index(drop=True)

    train_x = make_features(train_df.drop(columns=["Survived"]).copy(), feature_set)
    test_x = make_features(test_df.copy(), feature_set)

    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    cat_cols = [
        c
        for c in ["Sex", "Embarked", "Title", "CabinDeck", "TicketPrefix"]
        if c in combined.columns
    ]
    combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=False)

    x_train = combined.iloc[: len(train_x)].reset_index(drop=True)
    x_test = combined.iloc[len(train_x) :].reset_index(drop=True)

    return x_train, y, x_test
