import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "../data/raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "../data/processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# -----------------------
# Basic preprocessing utils
# -----------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill or drop missing values in the dataset."""
    for col in df.select_dtypes(include="number").columns:
        df[col].fillna(df[col].median(), inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    """Scale numerical features using StandardScaler."""
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df


# -----------------------
# Main workflow
# -----------------------
def load_data():
    """Load crop, soil, and weather datasets."""
    try:
        crop_df = pd.read_csv(os.path.join(RAW_DIR, "Crop_recommendation.csv"))
        soil_df = pd.read_csv(os.path.join(RAW_DIR, "data_core.csv"))
        weather_df = pd.read_csv(os.path.join(RAW_DIR, "weatherHistory.csv"))

        print("✅ Datasets loaded successfully!")
        return crop_df, soil_df, weather_df

    except FileNotFoundError as e:
        raise FileNotFoundError(f"❌ Missing file! Check RAW folder. Details: {e}")


def preprocess_and_merge():
    """Merge crop, soil, and weather data into a final dataset."""
    crop_df, soil_df, weather_df = load_data()

    # Merge Crop + Soil
    common_cols = list(set(crop_df.columns) & set(soil_df.columns))
    if common_cols:
        crop_soil_df = pd.merge(crop_df, soil_df, on=common_cols, how="inner")
        print(f"🔗 Crop + Soil merged on columns: {common_cols}")
    else:
        crop_soil_df = pd.concat([crop_df, soil_df], axis=1)
        print("⚠️ No common keys between Crop and Soil. Using concatenation.")

    # Merge with Weather
    common_weather_cols = list(set(crop_soil_df.columns) & set(weather_df.columns))
    if common_weather_cols:
        final_df = pd.merge(crop_soil_df, weather_df, on=common_weather_cols, how="left")
        print(f"🌦️ Crop+Soil + Weather merged on: {common_weather_cols}")
    else:
        final_df = crop_soil_df
        print("⚠️ No common keys with Weather dataset. Skipping merge.")

    # Handle missing values
    final_df = final_df.ffill().bfill().infer_objects(copy=False)

    # Save
    processed_file = os.path.join(PROCESSED_DIR, "final_dataset.csv")
    final_df.to_csv(processed_file, index=False)
    print(f"✅ Final dataset saved at: {processed_file}")

    return final_df


def load_and_prepare_data(path: str):
    """Load final dataset, clean it, scale features, and split X/y."""
    df = pd.read_csv(path)

    # Handle missing + scaling
    df = handle_missing_values(df)
    df = scale_features(df)

    # Ensure label column exists
    if "label" not in df.columns:
        raise KeyError("❌ 'label' column not found in dataset! Make sure preprocessing includes target labels.")

    X = df.drop("label", axis=1)
    y = df["label"]

    return X, y


# -----------------------
# Debug Run
# -----------------------
if __name__ == "__main__":
    # Merge datasets
    final_df = preprocess_and_merge()

    # Prepare features/labels
    processed_path = os.path.join(PROCESSED_DIR, "final_dataset.csv")
    X, y = load_and_prepare_data(processed_path)

    print("\n🔍 Preview of processed dataset:")
    print(final_df.head())
    print(f"\n✅ Features shape: {X.shape}, Labels shape: {y.shape}")
