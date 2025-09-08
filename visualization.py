"""
Visualization utilities.
Generate correlation heatmaps and feature importance plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def plot_correlation_heatmap(df: pd.DataFrame, save_path="outputs/reports/correlation_heatmap.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Features")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Correlation heatmap saved at {save_path}")


def plot_feature_importance(model, feature_names, save_path="outputs/reports/feature_importance.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    importances = model.feature_importances_
    sorted_idx = importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx], align="center")
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.title("Feature Importance")
    plt.savefig(save_path)
    plt.close()
    print(f"🌟 Feature importance plot saved at {save_path}")
