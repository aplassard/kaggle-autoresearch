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


QUALITY_MAPPING = {
    "Ex": 5,
    "Gd": 4,
    "TA": 3,
    "Fa": 2,
    "Po": 1,
}

QUALITY_COLUMNS = [
    "ExterQual",
    "BsmtQual",
    "HeatingQC",
    "KitchenQual",
    "GarageQual",
    "FireplaceQu",
]

BSMT_EXPOSURE_MAPPING = {
    "Gd": 4,
    "Av": 3,
    "Mn": 2,
    "No": 1,
}


def make_features(df: pd.DataFrame, feature_set: str = "baseline") -> pd.DataFrame:
    out = df.copy()

    if feature_set == "baseline":
        out["TotalSF"] = (
            out["TotalBsmtSF"].fillna(0)
            + out["1stFlrSF"].fillna(0)
            + out["2ndFlrSF"].fillna(0)
        )
        out["TotalBathrooms"] = (
            out["FullBath"].fillna(0)
            + out["HalfBath"].fillna(0) * 0.5
            + out["BsmtFullBath"].fillna(0)
            + out["BsmtHalfBath"].fillna(0) * 0.5
        )
        out["HouseAge"] = out["YrSold"].fillna(2010) - out["YearBuilt"].fillna(2000)
        out["RemodAge"] = out["YrSold"].fillna(2010) - out["YearRemodAdd"].fillna(2000)
        out["IsRemodeled"] = (out["YearBuilt"] != out["YearRemodAdd"]).astype(int)
        out["HasGarage"] = (out["GarageArea"].fillna(0) > 0).astype(int)
        out["HasFireplace"] = (out["Fireplaces"].fillna(0) > 0).astype(int)

        neighborhood_dummies = pd.get_dummies(
            out["Neighborhood"], prefix="Neighborhood", drop_first=True
        )
        out = pd.concat([out, neighborhood_dummies], axis=1)

        mszoning_dummies = pd.get_dummies(
            out["MSZoning"], prefix="MSZoning", drop_first=True
        )
        out = pd.concat([out, mszoning_dummies], axis=1)

        bldgtype_dummies = pd.get_dummies(
            out["BldgType"], prefix="BldgType", drop_first=True
        )
        out = pd.concat([out, bldgtype_dummies], axis=1)

        housestyle_dummies = pd.get_dummies(
            out["HouseStyle"], prefix="HouseStyle", drop_first=True
        )
        out = pd.concat([out, housestyle_dummies], axis=1)

        functional_dummies = pd.get_dummies(
            out["Functional"], prefix="Functional", drop_first=True
        )
        out = pd.concat([out, functional_dummies], axis=1)

        quality_encoded = []
        for col in QUALITY_COLUMNS:
            if col in out.columns:
                out[col + "_Ord"] = out[col].map(QUALITY_MAPPING).fillna(0)
                quality_encoded.append(col + "_Ord")

        if "BsmtExposure" in out.columns:
            out["BsmtExposure_Ord"] = (
                out["BsmtExposure"].map(BSMT_EXPOSURE_MAPPING).fillna(0)
            )
            bsmt_exposure_col = ["BsmtExposure_Ord"]
        else:
            bsmt_exposure_col = []

        out["OverallQual_Sq"] = out["OverallQual"] ** 2
        out["GarageCars_OverallQual"] = out["GarageCars"] * out["OverallQual"]
        out["BsmtQual_OverallQual"] = out["BsmtQual_Ord"] * out["OverallQual"]
        out["KitchenQual_OverallQual"] = out["KitchenQual_Ord"] * out["OverallQual"]
        out["FireplaceQu_OverallQual"] = out["FireplaceQu_Ord"] * out["OverallQual"]
        out["TotalSF_GarageCars"] = out["TotalSF"] * out["GarageCars"]
        out["GarageCars_KitchenQual_Ord"] = out["GarageCars"] * out["KitchenQual_Ord"]
        out["GarageCars_OverallQual_Sq"] = out["GarageCars"] * out["OverallQual_Sq"]
        out["GrLivArea_Ratio"] = out["GrLivArea"] / out["TotalSF"]

        cols = (
            NUMERIC_COLUMNS
            + [
                "TotalSF",
                "TotalBathrooms",
                "HouseAge",
                "RemodAge",
                "IsRemodeled",
                "HasGarage",
                "HasFireplace",
                "OverallQual_Sq",
                "GarageCars_OverallQual",
                "BsmtQual_OverallQual",
                "KitchenQual_OverallQual",
                "FireplaceQu_OverallQual",
                "TotalSF_GarageCars",
                "GarageCars_KitchenQual_Ord",
                "GarageCars_OverallQual_Sq",
                "GrLivArea_Ratio",
            ]
            + list(neighborhood_dummies.columns)
            + list(mszoning_dummies.columns)
            + list(bldgtype_dummies.columns)
            + list(housestyle_dummies.columns)
            + list(functional_dummies.columns)
            + quality_encoded
            + bsmt_exposure_col
        )
    else:
        cols = NUMERIC_COLUMNS

    available_cols = [c for c in cols if c in out.columns]
    return out[list(available_cols)]


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
