import pandas as pd
from sklearn.preprocessing import StandardScaler

NUMERIC_COLUMNS = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
]


def make_features(df: pd.DataFrame, feature_set: str = "baseline") -> pd.DataFrame:
    out = df.copy()

    if feature_set == "baseline":
        cols = NUMERIC_COLUMNS
    else:
        cols = NUMERIC_COLUMNS

    available_cols = [c for c in cols if c in out.columns]
    return out[available_cols]


def build_feature_matrices(train_df, test_df, feature_set="baseline"):
    y = train_df["SalePrice"].astype(float).reset_index(drop=True)

    train_x = make_features(train_df.drop(columns=["SalePrice"]).copy(), feature_set)
    test_x = make_features(test_df.copy(), feature_set)

    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    for col in combined.columns:
        combined[col] = combined[col].fillna(combined[col].median())

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    combined = pd.DataFrame(combined_scaled, columns=combined.columns)

    x_train = combined.iloc[: len(train_x)].reset_index(drop=True)
    x_test = combined.iloc[len(train_x) :].reset_index(drop=True)

    return x_train, y, x_test
