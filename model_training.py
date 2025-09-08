# src/model_training.py

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

def train_models():
    # ✅ Correct path to preprocessed CSV
    data_path = r"C:\Users\tusha\Downloads\Sustainable-Crop-Recommendation\data\preprocessed_crop_data.csv"
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset not found at {data_path}")
    print("✅ Preprocessed dataset found, loading...")

    # Load CSV directly
    df = pd.read_csv(data_path)

    # Handle missing values
    for col in df.select_dtypes(include='number').columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Separate features and target
    if "label" not in df.columns:
        raise KeyError("❌ CSV must contain a 'label' column as target.")
    X = df.drop("label", axis=1)
    y = df["label"]

    # Encode categorical columns
    encoders = {}
    for col in X.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    encoders["label"] = label_encoder

    # Scale numeric features
    numeric_cols = X.select_dtypes(include='number').columns
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # Save encoders
    os.makedirs("../outputs/models", exist_ok=True)
    joblib.dump(encoders, "../outputs/models/encoders.pkl")
    print("💾 Encoders saved to outputs/models/encoders.pkl")

    # Stratified split if feasible
    min_class_count = pd.Series(y).value_counts().min()
    stratify = y if min_class_count >= 2 else None
    if stratify is None:
        print("⚠️ Some classes <2 samples. Disabling stratification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )

    # Save splits
    splits_dir = os.path.join(os.path.dirname(__file__), "../data/splits")
    os.makedirs(splits_dir, exist_ok=True)
    pd.concat([X_train, pd.Series(y_train, name="label")], axis=1).to_csv(
        os.path.join(splits_dir, "train.csv"), index=False
    )
    pd.concat([X_test, pd.Series(y_test, name="label")], axis=1).to_csv(
        os.path.join(splits_dir, "test.csv"), index=False
    )
    print(f"💾 Train/Test splits saved in {splits_dir}")

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

    # Save evaluation report
    os.makedirs("../outputs/reports", exist_ok=True)
    report_path = "../outputs/reports/accuracy_report.txt"
    with open(report_path, "w") as f:
        for model, acc in results.items():
            f.write(f"{model}: {acc:.4f}\n")
        f.write("\nClassification Report (Random Forest):\n")
        f.write(classification_report(y_test, y_pred_rf))
    print(f"📊 Accuracy report saved at {report_path}")

    return results

if __name__ == "__main__":
    results = train_models()
    print("\n✅ Training complete. Results:")
    for model, acc in results.items():
        print(f"{model}: {acc:.4f}")
