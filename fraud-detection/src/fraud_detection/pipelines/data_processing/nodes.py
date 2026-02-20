import logging
import numpy as np
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)


def merge_datasets(
    transactions: pd.DataFrame, identity: pd.DataFrame
) -> pd.DataFrame:
    """Merge transactions and customers datasets on 'customer_id'."""
    logger.info(f"transactions shape: {transactions.shape}")
    logger.info(f"identity shape: {identity.shape}")
    logger.info("Merging transactions and identity datasets.")

    merged_df = pd.merge(transactions, identity, on="TransactionID", how="left")

    logger.info(f"Merged dataset shape: {merged_df.shape}")
    return merged_df


def engineer_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Engineer new features for fraud detection."""
    logger.info("Engineering new features.")
    df = df.copy()

    night_start = params["night_hours_start"]
    night_end = params["night_hours_end"]
    card_col = params["card_group_column"]

    df["hour"] = (df["TransactionDT"] // 3600) % 24
    df["day_of_week"] = (df["TransactionDT"] // (3600 * 24)) % 7
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_night"] = ((df["hour"] >= night_start) | (df["hour"] < night_end)).astype(int)

    df["amt_log"] = np.log1p(df["TransactionAmt"])
    df["amt_decimal"] = df["TransactionAmt"] % 1
    df["is_round_amt"] = (df["amt_decimal"] == 0).astype(int)

    df["P_email_provider"] = df["P_emaildomain"].apply(lambda x: str(x).split(".")[0] if isinstance(x, str) else "unknown")
    df["R_email_provider"] = df["R_emaildomain"].apply(lambda x: str(x).split(".")[0] if isinstance(x, str) else "unknown")

    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype(int)

    df["card_id"] = (
        df["card1"].astype(str) + "_" +
        df["card2"].astype(str) + "_" +
        df["card3"].astype(str) + "_" +
        df["card4"].astype(str) + "_" +
        df["card5"].astype(str) + "_" +
        df["card6"].astype(str)
    )

    card_freq = df.groupby(card_col)["TransactionID"].agg(
        card_txn_count="count",
        card_amt_mean="mean",
        card_amt_std="std"
        ).reset_index()
    
    card_freq.columns = [card_col, "card_txn_count", "card_amt_mean", "card_amt_std"]

    df = df.merge(card_freq, on=card_col, how="left")

    logger.info("Feature engineering completed.")
    return df


def handle_missing_values(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    logger.info("Handling missing values.")
    df = df.copy()

    threshold = params.get("missing_threshold", 0.9)
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index

    df = df.drop(columns=cols_to_drop.tolist())

    logger.info(f"Dropped columns with more than {threshold*100}% missing values: {cols_to_drop.tolist()}")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].median())

    logger.info("Missing values handled.")
    logger.info(f"Final dataset shape after handling missing values: {df.shape}")
    return df



def encode_categorical_variables(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Encode categorical variables using one-hot encoding."""
    logger.info("Encoding categorical variables.")
    df = df.copy()

    categorical_cols = df.select_dtypes(include=["object"]).columns.to_list()

    exclude_cols = params.get("encoding_exclude", [])
    categorical_cols = [col for col in categorical_cols if col not in exclude_cols]

    for col in categorical_cols:
        if df[col].nunique() > 10:
            top_categories = df[col].value_counts().nlargest(10).index
            df[col] = df[col].where(df[col].isin(top_categories), other="other")

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    logger.info("Categorical variables encoded.")
    logger.info(f"Final dataset shape after encoding: {df.shape}")
    return df


def split_features_target(df: pd.DataFrame, params: dict) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataset into features and target variable."""
    logger.info("Splitting features and target variable.")
    target = params.get("target_column", "isFraud")
    drop_cols = params.get("drop_columns", [])
    drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target]
    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")
    return X, y