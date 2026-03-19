import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent


def make_features(
    df: pd.DataFrame,
    stores_df: pd.DataFrame,
    oil_df: pd.DataFrame,
    holidays_df: pd.DataFrame,
    feature_set: str = "baseline",
    family_encoder: LabelEncoder = None,
    city_encoder: LabelEncoder = None,
    state_encoder: LabelEncoder = None,
    cluster_encoder: LabelEncoder = None,
    store_type_encoder: LabelEncoder = None,
) -> pd.DataFrame:
    out = df.copy()

    out = out.merge(stores_df, on="store_nbr", how="left")

    oil_df_clean = oil_df.dropna(subset=["dcoilwtico"]).copy()
    oil_df_clean = oil_df_clean.rename(columns={"dcoilwtico": "oil_price"})
    out = out.merge(oil_df_clean[["date", "oil_price"]], on="date", how="left")
    out["oil_price"] = out["oil_price"].ffill().bfill()

    if feature_set == "baseline":
        out["year"] = out["date"].dt.year
        out["month"] = out["date"].dt.month
        out["day"] = out["date"].dt.day
        out["dayofweek"] = out["date"].dt.dayofweek
        out["dayofyear"] = out["date"].dt.dayofyear
        out["weekofyear"] = out["date"].dt.isocalendar().week.astype(int)
        out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
        out["is_month_start"] = out["date"].dt.is_month_start.astype(int)
        out["is_month_end"] = out["date"].dt.is_month_end.astype(int)
        out["is_payday"] = (
            (out["day"] == 15) | (out["day"] == out["date"].dt.days_in_month)
        ).astype(int)

        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
        out["dayofweek_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
        out["dayofweek_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
        out["dayofyear_sin"] = np.sin(2 * np.pi * out["dayofyear"] / 365)
        out["dayofyear_cos"] = np.cos(2 * np.pi * out["dayofyear"] / 365)
        out["weekofyear_sin"] = np.sin(2 * np.pi * out["weekofyear"] / 52)
        out["weekofyear_cos"] = np.cos(2 * np.pi * out["weekofyear"] / 52)
        out["day_sin"] = np.sin(2 * np.pi * out["day"] / 31)
        out["day_cos"] = np.cos(2 * np.pi * out["day"] / 31)

        out["is_quake_period"] = (
            (out["date"] >= "2016-04-16") & (out["date"] <= "2016-05-31")
        ).astype(int)

        out["store_nbr"] = out["store_nbr"].astype(int)

        if family_encoder is not None:
            out["family_enc"] = family_encoder.transform(out["family"])
        else:
            out["family_enc"] = 0

        if city_encoder is not None:
            out["city_enc"] = city_encoder.transform(out["city"])
        else:
            out["city_enc"] = 0

        if state_encoder is not None:
            out["state_enc"] = state_encoder.transform(out["state"])
        else:
            out["state_enc"] = 0

        if cluster_encoder is not None:
            out["cluster_enc"] = cluster_encoder.transform(out["cluster"].astype(str))
        else:
            out["cluster_enc"] = 0

        if store_type_encoder is not None:
            out["store_type_enc"] = store_type_encoder.transform(out["type"])
        else:
            out["store_type_enc"] = 0

        out["onpromotion"] = out["onpromotion"].fillna(0)

        holiday_local = holidays_df[
            (holidays_df["transferred"] == False) & (holidays_df["locale"] == "Local")
        ][["date", "locale_name"]].drop_duplicates()
        holiday_local["is_local_holiday"] = 1
        holiday_local = holiday_local.rename(columns={"locale_name": "city"})
        out = out.merge(holiday_local, on=["date", "city"], how="left")
        out["is_local_holiday"] = out["is_local_holiday"].fillna(0).astype(int)

        holiday_national = holidays_df[
            (holidays_df["transferred"] == False)
            & (holidays_df["locale"] == "National")
        ][["date"]].drop_duplicates()
        holiday_national["is_national_holiday"] = 1
        out = out.merge(holiday_national, on="date", how="left")
        out["is_national_holiday"] = out["is_national_holiday"].fillna(0).astype(int)

        oil_unique = out[["date", "oil_price"]].drop_duplicates().sort_values("date")
        oil_unique = oil_unique.set_index("date")
        oil_unique["oil_lag_1"] = oil_unique["oil_price"].shift(1)
        oil_unique["oil_lag_7"] = oil_unique["oil_price"].shift(7)
        oil_unique["oil_lag_14"] = oil_unique["oil_price"].shift(14)
        oil_unique["oil_roll_mean_7"] = (
            oil_unique["oil_price"].rolling(window=7, min_periods=1).mean()
        )
        oil_unique["oil_roll_mean_14"] = (
            oil_unique["oil_price"].rolling(window=14, min_periods=1).mean()
        )
        oil_unique["oil_roll_mean_30"] = (
            oil_unique["oil_price"].rolling(window=30, min_periods=1).mean()
        )
        oil_unique["oil_roll_std_7"] = (
            oil_unique["oil_price"].rolling(window=7, min_periods=1).std()
        )
        oil_unique["oil_roll_std_14"] = (
            oil_unique["oil_price"].rolling(window=14, min_periods=1).std()
        )
        oil_unique["oil_roll_std_30"] = (
            oil_unique["oil_price"].rolling(window=30, min_periods=1).std()
        )
        oil_unique = oil_unique.reset_index()
        out = out.drop(
            columns=[
                "oil_lag_1",
                "oil_lag_7",
                "oil_lag_14",
                "oil_roll_mean_7",
                "oil_roll_mean_14",
                "oil_roll_mean_30",
                "oil_roll_std_7",
                "oil_roll_std_14",
                "oil_roll_std_30",
            ],
            errors="ignore",
        )
        out = out.merge(oil_unique, on="date", how="left", suffixes=("", "_y"))
        out["oil_price"] = (
            out["oil_price"].fillna(out["oil_price_y"]).fillna(out["oil_price"])
        )
        out = out.drop(columns=["oil_price_y"], errors="ignore")

        cols = [
            "store_nbr",
            "family_enc",
            "onpromotion",
            "year",
            "month",
            "day",
            "dayofweek",
            "dayofyear",
            "weekofyear",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "is_payday",
            "is_quake_period",
            "city_enc",
            "state_enc",
            "cluster_enc",
            "store_type_enc",
            "oil_price",
            "is_local_holiday",
            "is_national_holiday",
            "oil_lag_1",
            "oil_lag_7",
            "oil_lag_14",
            "oil_roll_mean_7",
            "oil_roll_mean_14",
            "oil_roll_mean_30",
            "oil_roll_std_7",
            "oil_roll_std_14",
            "oil_roll_std_30",
            "month_sin",
            "month_cos",
            "dayofweek_sin",
            "dayofweek_cos",
            "dayofyear_sin",
            "dayofyear_cos",
            "weekofyear_sin",
            "weekofyear_cos",
            "day_sin",
            "day_cos",
        ]
    else:
        cols = ["store_nbr", "onpromotion"]

    available_cols = [c for c in cols if c in out.columns]
    return out[available_cols], out["date"]


def build_feature_matrices(
    train_df, test_df, stores_df, oil_df, holidays_df, feature_set="baseline"
):
    transactions_df = pd.read_csv(
        ROOT / "data" / "transactions.csv", parse_dates=["date"]
    )

    family_encoder = LabelEncoder()
    family_encoder.fit(train_df["family"])

    city_encoder = LabelEncoder()
    city_encoder.fit(stores_df["city"])

    state_encoder = LabelEncoder()
    state_encoder.fit(stores_df["state"])

    cluster_encoder = LabelEncoder()
    cluster_encoder.fit(stores_df["cluster"].astype(str))

    store_type_encoder = LabelEncoder()
    store_type_encoder.fit(stores_df["type"])

    train_x, train_dates = make_features(
        train_df.drop(columns=["sales"]),
        stores_df,
        oil_df,
        holidays_df,
        feature_set,
        family_encoder,
        city_encoder,
        state_encoder,
        cluster_encoder,
        store_type_encoder,
    )
    test_x, _ = make_features(
        test_df,
        stores_df,
        oil_df,
        holidays_df,
        feature_set,
        family_encoder,
        city_encoder,
        state_encoder,
        cluster_encoder,
        store_type_encoder,
    )

    y = train_df["sales"].astype(float).reset_index(drop=True)

    train_df_sorted = train_df.sort_values(["store_nbr", "family", "date"]).copy()
    train_df_sorted["sales_lag_7"] = train_df_sorted.groupby(["store_nbr", "family"])[
        "sales"
    ].shift(7)
    train_df_sorted["sales_lag_16"] = train_df_sorted.groupby(["store_nbr", "family"])[
        "sales"
    ].shift(16)
    train_df_sorted["sales_lag_21"] = train_df_sorted.groupby(["store_nbr", "family"])[
        "sales"
    ].shift(21)
    train_df_sorted["sales_lag_28"] = train_df_sorted.groupby(["store_nbr", "family"])[
        "sales"
    ].shift(28)
    train_df_sorted["sales_lag_35"] = train_df_sorted.groupby(["store_nbr", "family"])[
        "sales"
    ].shift(35)
    train_df_sorted["sales_lag_60"] = train_df_sorted.groupby(["store_nbr", "family"])[
        "sales"
    ].shift(60)
    train_df_sorted["sales_roll_mean_7"] = (
        train_df_sorted.groupby(["store_nbr", "family"])["sales"]
        .shift(16)
        .rolling(window=7, min_periods=1)
        .mean()
    )
    train_df_sorted["sales_roll_mean_14"] = (
        train_df_sorted.groupby(["store_nbr", "family"])["sales"]
        .shift(16)
        .rolling(window=14, min_periods=1)
        .mean()
    )
    train_df_sorted["sales_roll_mean_30"] = (
        train_df_sorted.groupby(["store_nbr", "family"])["sales"]
        .shift(16)
        .rolling(window=30, min_periods=1)
        .mean()
    )

    last_train_date = train_df["date"].max()
    sales_lag_map = (
        train_df_sorted[train_df_sorted["date"] == last_train_date][
            [
                "store_nbr",
                "family",
                "sales_lag_7",
                "sales_lag_16",
                "sales_lag_21",
                "sales_lag_28",
                "sales_lag_35",
                "sales_lag_60",
                "sales_roll_mean_7",
                "sales_roll_mean_14",
                "sales_roll_mean_30",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    train_lags = train_df_sorted[
        [
            "id",
            "sales_lag_7",
            "sales_lag_16",
            "sales_lag_21",
            "sales_lag_28",
            "sales_lag_35",
            "sales_lag_60",
            "sales_roll_mean_7",
            "sales_roll_mean_14",
            "sales_roll_mean_30",
        ]
    ].copy()
    train_x = train_x.reset_index(drop=True)
    train_lags = train_lags.set_index("id")
    train_x = train_x.join(
        train_lags, on=train_df["id"].values, how="left", rsuffix="_lag"
    )

    test_x = test_x.reset_index(drop=True)
    test_merged = test_df[["id", "store_nbr", "family"]].copy()
    test_merged = test_merged.merge(
        sales_lag_map,
        on=["store_nbr", "family"],
        how="left",
    )
    for col in [
        "sales_lag_7",
        "sales_lag_16",
        "sales_lag_21",
        "sales_lag_28",
        "sales_lag_35",
        "sales_lag_60",
        "sales_roll_mean_7",
        "sales_roll_mean_14",
        "sales_roll_mean_30",
    ]:
        test_x[col] = test_merged[col].values

    trans_sorted = transactions_df.sort_values(["store_nbr", "date"]).copy()
    trans_sorted["trans_lag_16"] = trans_sorted.groupby("store_nbr")[
        "transactions"
    ].shift(16)
    trans_sorted["trans_lag_21"] = trans_sorted.groupby("store_nbr")[
        "transactions"
    ].shift(21)
    trans_sorted["trans_lag_28"] = trans_sorted.groupby("store_nbr")[
        "transactions"
    ].shift(28)
    trans_sorted["trans_roll_mean_7"] = (
        trans_sorted.groupby("store_nbr")["transactions"]
        .shift(16)
        .rolling(window=7, min_periods=1)
        .mean()
    )
    trans_sorted["trans_roll_mean_14"] = (
        trans_sorted.groupby("store_nbr")["transactions"]
        .shift(16)
        .rolling(window=14, min_periods=1)
        .mean()
    )
    trans_sorted["trans_roll_mean_30"] = (
        trans_sorted.groupby("store_nbr")["transactions"]
        .shift(16)
        .rolling(window=30, min_periods=1)
        .mean()
    )

    last_trans_date = transactions_df["date"].max()
    trans_lag_map = (
        trans_sorted[trans_sorted["date"] == last_trans_date][
            [
                "store_nbr",
                "trans_lag_16",
                "trans_lag_21",
                "trans_lag_28",
                "trans_roll_mean_7",
                "trans_roll_mean_14",
                "trans_roll_mean_30",
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    train_trans = train_df[["id", "store_nbr", "date"]].copy()
    train_trans = train_trans.merge(
        trans_sorted[
            [
                "store_nbr",
                "date",
                "trans_lag_16",
                "trans_lag_21",
                "trans_lag_28",
                "trans_roll_mean_7",
                "trans_roll_mean_14",
                "trans_roll_mean_30",
            ]
        ],
        on=["store_nbr", "date"],
        how="left",
    )
    for col in [
        "trans_lag_16",
        "trans_lag_21",
        "trans_lag_28",
        "trans_roll_mean_7",
        "trans_roll_mean_14",
        "trans_roll_mean_30",
    ]:
        train_x[col] = train_trans[col].values

    test_trans = test_df[["id", "store_nbr"]].copy()
    test_trans = test_trans.merge(trans_lag_map, on="store_nbr", how="left")
    for col in [
        "trans_lag_16",
        "trans_lag_21",
        "trans_lag_28",
        "trans_roll_mean_7",
        "trans_roll_mean_14",
        "trans_roll_mean_30",
    ]:
        test_x[col] = test_trans[col].values

    store_family_stats = (
        train_df.groupby(["store_nbr", "family"])["sales"]
        .agg(["mean", "std"])
        .reset_index()
    )
    store_family_stats.columns = [
        "store_nbr",
        "family",
        "store_family_mean",
        "store_family_std",
    ]
    store_family_stats["store_family_std"] = store_family_stats[
        "store_family_std"
    ].fillna(0)

    train_stats = train_df[["id", "store_nbr", "family"]].copy()
    train_stats = train_stats.merge(
        store_family_stats, on=["store_nbr", "family"], how="left"
    )
    train_x["store_family_mean"] = train_stats["store_family_mean"].values
    train_x["store_family_std"] = train_stats["store_family_std"].values

    test_stats = test_df[["id", "store_nbr", "family"]].copy()
    test_stats = test_stats.merge(
        store_family_stats, on=["store_nbr", "family"], how="left"
    )
    test_x["store_family_mean"] = test_stats["store_family_mean"].values
    test_x["store_family_std"] = test_stats["store_family_std"].values

    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

    for col in combined.columns:
        combined[col] = combined[col].fillna(combined[col].median())

    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    combined = pd.DataFrame(combined_scaled, columns=combined.columns)

    x_train = combined.iloc[: len(train_x)].reset_index(drop=True)
    x_test = combined.iloc[len(train_x) :].reset_index(drop=True)

    return x_train, y, x_test, train_dates.reset_index(drop=True)
