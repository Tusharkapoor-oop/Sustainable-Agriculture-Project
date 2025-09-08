"""
Model evaluation utilities.
Generate reports, confusion matrices, classification reports, and save them.
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)


def evaluate_model(model, X_test, y_test, label_encoder, model_name: str):
    """
    Evaluate model performance and save confusion matrix + report.
    """
    outputs_dir = os.path.join("outputs", "reports")
    os.makedirs(outputs_dir, exist_ok=True)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save classification report
    report_path = os.path.join(outputs_dir, f"{model_name}_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(
            classification_report(
                y_test,
                y_pred,
                target_names=label_encoder.classes_,
            )
        )

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    cm_path = os.path.join(outputs_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"✅ {model_name} evaluated. Accuracy: {acc:.4f}")
    print(f"📊 Reports saved to {outputs_dir}")

    return acc
