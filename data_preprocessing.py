import os
import pandas as pd

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "../data/raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "../data/processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_data():
    """Load crop, soil, and weather datasets"""
    try:
        crop_df = pd.read_csv(os.path.join(RAW_DIR, "Crop_recommendation.csv"))
        soil_df = pd.read_csv(os.path.join(RAW_DIR, "data_core.csv"))
        weather_df = pd.read_csv(os.path.join(RAW_DIR, "weatherHistory.csv"))
        print(" Datasets loaded successfully!")
        return crop_df, soil_df, weather_df
    except FileNotFoundError as e:
        raise FileNotFoundError(f" Missing file! Check RAW folder. Details: {e}")

def preprocess():
    crop_df, soil_df, weather_df = load_data()

    # --- Merge Crop + Soil on N, P, K ---
    common_cols = list(set(crop_df.columns) & set(soil_df.columns))
    if common_cols:
        crop_soil_df = pd.merge(crop_df, soil_df, on=common_cols, how="inner")
        print(f" Crop + Soil merged on columns: {common_cols}")
    else:
        crop_soil_df = pd.concat([crop_df, soil_df], axis=1)
        print(" No common keys found between Crop and Soil. Using concatenation.")

    # --- Merge with Weather dataset (on Humidity or other shared cols) ---
    common_weather_cols = list(set(crop_soil_df.columns) & set(weather_df.columns))
    if common_weather_cols:
        final_df = pd.merge(crop_soil_df, weather_df, on=common_weather_cols, how="left")
        print(f" Crop+Soil + Weather merged on columns: {common_weather_cols}")
    else:
        final_df = crop_soil_df
        print(" No common keys found with Weather dataset. Skipping merge.")

    # --- Handle missing values safely ---
    final_df = final_df.ffill().bfill()  # forward + backward fill
    final_df = final_df.infer_objects(copy=False)  # prevent future warnings

    # --- Save final dataset ---
    processed_file = os.path.join(PROCESSED_DIR, "final_dataset.csv")
    final_df.to_csv(processed_file, index=False)
    print(f" Final dataset saved at: {processed_file}")

if __name__ == "__main__":
    preprocess()
