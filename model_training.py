# src/model_training.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_models():
    # ✅ Correct path handling
    data_path = os.path.join(os.path.dirname(__file__), "../data/preprocessed_crop_data.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at {data_path}")

    print("✅ Processed dataset loaded successfully!")
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop("label", axis=1)
    y = df["label"]

    # Encode categorical columns (both X and y)
    encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le  # save encoder

    # Encode target labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    encoders["label"] = label_encoder

    # Save encoders for later use in prediction
    os.makedirs("../outputs/models", exist_ok=True)
    joblib.dump(encoders, "../outputs/models/encoders.pkl")
    print("💾 Encoders saved to outputs/models/encoders.pkl")

    # Check stratification feasibility
    min_class_count = pd.Series(y).value_counts().min()
    stratify = y if min_class_count >= 2 else None
    if stratify is None:
        print("⚠️ Some classes <2 samples. Disabling stratification.")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    results = {}

    # 1️⃣ Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results["RandomForest"] = acc_rf
    joblib.dump(rf, "../outputs/models/random_forest.pkl")
    print(f"🌳 Random Forest Accuracy: {acc_rf:.4f}")

        # 2️⃣ SVM
    svm = SVC(kernel="rbf", probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    results["SVM"] = acc_svm
    joblib.dump(svm, "../outputs/models/svm_model.pkl")
    print(f"📈 SVM Accuracy: {acc_svm:.4f}")

    # ✅ Ensure reports directory exists
    os.makedirs("../outputs/reports", exist_ok=True)

    # ✅ Save evaluation report
    report_path = "../outputs/reports/accuracy_report.txt"
    with open(report_path, "w") as f:
        for model, acc in results.items():
            f.write(f"{model}: {acc:.4f}\n")
        f.write("\nClassification Report (Random Forest):\n")
        f.write(classification_report(y_test, y_pred_rf))

    print(f"📊 Accuracy report saved at {report_path}")

    return results   # <-- don’t forget to return results


if __name__ == "__main__":
    results = train_models()
    print("\n✅ Training complete. Results:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")
