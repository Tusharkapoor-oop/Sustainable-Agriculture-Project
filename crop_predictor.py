import os
import pandas as pd
import joblib

# Global variable to hold models (optional, keeps memory efficient)
MODELS = None

def load_models(models_path: str = r"C:\Users\tusha\Downloads\Sustainable-Crop-Recommendation\outputs\models"):
    """
    Load trained models and encoders.
    Returns a dict with Random Forest, SVM, and encoders.
    """
    models = {
        "random_forest": joblib.load(os.path.join(models_path, "random_forest.pkl")),
        "svm": joblib.load(os.path.join(models_path, "svm_model.pkl")),
        "encoders": joblib.load(os.path.join(models_path, "encoders.pkl")),
    }
    return models

def preprocess_input(user_input: dict, encoders: dict) -> pd.DataFrame:
    """
    Convert user input dict to DataFrame and encode categorical features
    """
    # Convert keys to lowercase to match training feature names
    user_input = {k.lower(): v for k, v in user_input.items()}
    df = pd.DataFrame([user_input])

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col != "label" and col in df.columns:
            df[col] = le.transform(df[col])
    return df

def predict_crop(user_input: dict, models: dict) -> dict:
    """
    Predict crop recommendation using all trained models (RandomForest, SVM).
    
    Args:
        user_input (dict): Dictionary of farm features (N, P, K, temperature, humidity, ph, rainfall)
        models (dict): Dictionary containing trained models and encoders

    Returns:
        dict: { "random_forest": "rice", "svm": "wheat" }
    """
    # Preprocess input using saved encoders
    X = preprocess_input(user_input, models["encoders"])

    predictions = {}
    for model_name, model in models.items():
        if model_name == "encoders":
            continue  # skip encoders, not a model
        pred_index = model.predict(X)[0]
        crop = models["encoders"]["label"].inverse_transform([pred_index])[0]
        predictions[model_name] = crop

    return predictions


if __name__ == "__main__":
    # Load models once
    models = load_models()

    # Example input
    sample_input = {
        "n": 90,
        "p": 42,
        "k": 43,
        "temperature": 20.8,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 200,
    }

    results = predict_crop(sample_input, models)
    for model_name, crop in results.items():
        print(f"🌱 Recommended Crop ({model_name}): {crop}")
