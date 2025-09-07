import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Encode categorical columns
# --------------------------
def encode_categorical(df):
    label_encoders = {}
    for col in df.columns:
        if df[col].dtype == 'object':  # if column is categorical (string)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    return df, label_encoders

# --------------------------
# Train and evaluate models
# --------------------------
def train_models():
    # Load dataset
    df = pd.read_csv("data/processed_dataset.csv")
    print("Processed dataset loaded successfully!")

    # Encode categorical columns
    df, encoders = encode_categorical(df)

    # Split features/labels
    X = df.drop("label", axis=1)
    y = df["label"]

    # Handle case when some classes have <2 samples
    stratify_option = y if y.value_counts().min() >= 2 else None
    if stratify_option is None:
        print("⚠️ Some classes have <2 samples. Disabling stratification for splitting.")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_option
    )

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel="linear", probability=True, random_state=42),
    }

    results = {}
    for name, model in models.items():
        print(f"\n🔹 Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"{name} Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds))

        # Save model
        joblib.dump(model, f"models/{name}_model.pkl")
        results[name] = acc

    return results

if __name__ == "__main__":
    results = train_models()
    print("\n✅ Training complete! Results:", results)
