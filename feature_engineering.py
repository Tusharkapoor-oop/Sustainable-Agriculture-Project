"""
Feature engineering utilities.
Adds extra sustainability-focused features like soil fertility index or drought score.
"""

import pandas as pd


def add_soil_fertility_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a soil fertility index = weighted sum of N, P, K.
    """
    df["soil_fertility_index"] = (
        0.4 * df["n"] + 0.3 * df["p"] + 0.3 * df["k"]
    )
    return df


def add_drought_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add drought score based on rainfall & temperature.
    Higher temperature & lower rainfall → higher drought score.
    """
    df["drought_score"] = (
        (df["temperature"] / df["temperature"].max()) * 0.6
        + (1 - df["rainfall"] / df["rainfall"].max()) * 0.4
    )
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering functions.
    """
    df = add_soil_fertility_index(df)
    df = add_drought_score(df)
    return df
