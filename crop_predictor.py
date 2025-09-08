import os
import pandas as pd
import joblib

# Load trained models and encoders
models_path = "../outputs/models"
rf_model = joblib.load(os.path.join(models_path, "random_forest.pkl"))
svm_model = joblib.load(os.path.join(models_path, "svm_model.pkl"))
encoders = joblib.load(os.path.join(models_path, "encoders.pkl"))

def preprocess_input(user_input: dict) -> pd.DataFrame:
    """
    Convert user input dict to DataFrame and encode categorical features
    """
    # Convert all keys to lowercase to match training feature names
    user_input = {k.lower(): v for k, v in user_input.items()}
    
    df = pd.DataFrame([user_input])

    # Encode categorical columns using saved encoders
    for col, le in encoders.items():
        if col != "label" and col in df.columns:
            df[col] = le.transform(df[col])
    return df

def predict_crop(user_input: dict):
    """
    Predict crop recommendation using both Random Forest and SVM models
    """
    X = preprocess_input(user_input)

    # Random Forest prediction
    rf_pred_index = rf_model.predict(X)[0]
    rf_crop = encoders["label"].inverse_transform([rf_pred_index])[0]

    # SVM prediction
    svm_pred_index = svm_model.predict(X)[0]
    svm_crop = encoders["label"].inverse_transform([svm_pred_index])[0]

    return rf_crop, svm_crop

if __name__ == "__main__":
    # Example input (make sure keys match training data, lowercase)
    sample_input = {
        "n": 90,
        "p": 42,
        "k": 43,
        "temperature": 20.8,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 200
    }

    rf_crop, svm_crop = predict_crop(sample_input)

    print(f"🌱 Recommended Crop (Random Forest): {rf_crop}")
    print(f"🌾 Recommended Crop (SVM): {svm_crop}")
