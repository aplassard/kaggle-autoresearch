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


def make_features(
    df: pd.DataFrame,
    feature_set: str = "baseline",
    neighborhood_target_encoding: dict = None,
) -> pd.DataFrame:
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
        out["HasPool"] = (out["PoolArea"].fillna(0) > 0).astype(int)

        neighborhood_dummies = pd.get_dummies(
            out["Neighborhood"], prefix="Neighborhood", drop_first=True
        )
        out = pd.concat([out, neighborhood_dummies], axis=1)

        target_enc_cols = []
        if neighborhood_target_encoding is not None and "Neighborhood" in out.columns:
            global_mean = neighborhood_target_encoding.get("_global_mean_", 180000)
            out["Neighborhood_TargetEnc"] = (
                out["Neighborhood"]
                .map(neighborhood_target_encoding)
                .fillna(global_mean)
            )
            target_enc_cols = ["Neighborhood_TargetEnc"]

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

        out["BsmtQualityIndex"] = out["BsmtQual_Ord"] * out["BsmtExposure_Ord"]
        out["OverallQual_Sq"] = out["OverallQual"] ** 2
        out["GarageCars_OverallQual"] = out["GarageCars"] * out["OverallQual"]
        out["BsmtQual_OverallQual"] = out["BsmtQual_Ord"] * out["OverallQual"]
        out["KitchenQual_OverallQual"] = out["KitchenQual_Ord"] * out["OverallQual"]
        out["FireplaceQu_OverallQual"] = out["FireplaceQu_Ord"] * out["OverallQual"]
        out["TotalSF_GarageCars"] = out["TotalSF"] * out["GarageCars"]
        out["GarageCars_KitchenQual_Ord"] = out["GarageCars"] * out["KitchenQual_Ord"]
        out["GarageCars_OverallQual_Sq"] = out["GarageCars"] * out["OverallQual_Sq"]
        out["GrLivArea_Ratio"] = out["GrLivArea"] / out["TotalSF"]
        out["BsmtLivArea_Ratio"] = out["TotalBsmtSF"].fillna(0) / out["GrLivArea"]
        out["IsPeakSeason"] = out["MoSold"].isin([5, 6, 7]).astype(int)
        out["HasMultipleFireplaces"] = (out["Fireplaces"].fillna(0) >= 2).astype(int)
        out["HasHighQualityKitchen"] = (out["KitchenQual_Ord"] >= 4).astype(int)
        out["TotalSF_OverallCond"] = out["TotalSF"] * out["OverallCond"]
        out["IsRemodeled_OverallQual"] = out["IsRemodeled"] * out["OverallQual"]
        out["TotalSF_IsRemodeled"] = out["TotalSF"] * out["IsRemodeled"]
        out["GarageCars_BsmtQualityIndex"] = out["GarageCars"] * out["BsmtQualityIndex"]
        out["BathBedroomRatio"] = out["TotalBathrooms"] / out["BedroomAbvGr"].replace(
            0, 1
        )

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
                "HasPool",
                "BsmtQualityIndex",
                "OverallQual_Sq",
                "GarageCars_OverallQual",
                "BsmtQual_OverallQual",
                "KitchenQual_OverallQual",
                "FireplaceQu_OverallQual",
                "TotalSF_GarageCars",
                "GarageCars_KitchenQual_Ord",
                "GarageCars_OverallQual_Sq",
                "GrLivArea_Ratio",
                "BsmtLivArea_Ratio",
                "IsPeakSeason",
                "HasMultipleFireplaces",
                "HasHighQualityKitchen",
                "TotalSF_OverallCond",
                "IsRemodeled_OverallQual",
                "TotalSF_IsRemodeled",
                "GarageCars_BsmtQualityIndex",
                "BathBedroomRatio",
                "Neighborhood_TargetEnc",
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

    global_mean = y.mean()
    neighborhood_means = train_df.groupby("Neighborhood")["SalePrice"].mean()
    neighborhood_target_encoding = {"_global_mean_": global_mean}
    neighborhood_target_encoding.update(neighborhood_means.to_dict())

    train_x = make_features(
        train_df.drop(columns=["SalePrice"]).copy(),
        feature_set,
        neighborhood_target_encoding=neighborhood_target_encoding,
    )
    test_x = make_features(
        test_df.copy(),
        feature_set,
        neighborhood_target_encoding=neighborhood_target_encoding,
    )

    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    for col in combined.columns:
        combined[col] = combined[col].fillna(combined[col].median())

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    combined = pd.DataFrame(combined_scaled, columns=combined.columns)

    x_train = combined.iloc[: len(train_x)].reset_index(drop=True)
    x_test = combined.iloc[len(train_x) :].reset_index(drop=True)

    return x_train, y, x_test
