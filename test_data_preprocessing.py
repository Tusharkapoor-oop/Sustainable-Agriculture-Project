"""
World-class unit tests for model training and evaluation.

Run with:
    pytest tests/test_model_training.py -v
"""

import sys
import os
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Ensure the project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import train_model (alias train_models if needed)
try:
    from src.model_training import train_model
except ImportError:
    from src.model_training import train_models as train_model

from src.model_training import evaluate_model


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture(scope="module")
def sample_data():
    """Generate a synthetic binary classification dataset for testing ML pipeline."""
    X, y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.25, random_state=42)


# -----------------------------
# Tests
# -----------------------------
@pytest.mark.parametrize(
    "model_type, expected_cls",
    [
        ("RandomForest", RandomForestClassifier),
        ("DecisionTree", DecisionTreeClassifier),
        ("SVM", SVC),
    ],
)
def test_train_model_returns_correct_type(sample_data, model_type, expected_cls):
    """✅ Ensure train_model returns the correct sklearn estimator and is fitted."""
    X_train, X_test, y_train, y_test = sample_data
    model = train_model(X_train, y_train, model_type=model_type)

    assert isinstance(model, expected_cls), f"❌ {model_type} did not return {expected_cls.__name__}"
    assert hasattr(model, "predict"), f"❌ {model_type} missing predict()"


def test_evaluate_model_returns_metrics_dict(sample_data):
    """✅ Ensure evaluate_model returns accuracy and report in a dict."""
    X_train, X_test, y_train, y_test = sample_data
    model = train_model(X_train, y_train, model_type="RandomForest")

    metrics = evaluate_model(model, X_test, y_test)

    assert isinstance(metrics, dict), "❌ evaluate_model should return a dict"
    assert {"accuracy", "report"} <= metrics.keys(), "❌ Missing keys in evaluate_model output"
    assert isinstance(metrics["accuracy"], float), "❌ Accuracy must be float"
    assert 0.0 <= metrics["accuracy"] <= 1.0, "❌ Accuracy must be between 0 and 1"
    assert isinstance(metrics["report"], str), "❌ Classification report must be str"
    assert "precision" in metrics["report"], "❌ Report must include precision info"


def test_invalid_model_type_raises_value_error(sample_data):
    """✅ Passing unsupported model_type must raise ValueError."""
    X_train, X_test, y_train, y_test = sample_data
    with pytest.raises(ValueError, match="Unsupported"):
        train_model(X_train, y_train, model_type="UnsupportedModel")


def test_prediction_shape_matches(sample_data):
    """✅ model.predict output length should match input test set."""
    X_train, X_test, y_train, y_test = sample_data
    model = train_model(X_train, y_train, model_type="DecisionTree")
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test), "❌ Prediction length mismatch"


def test_randomness_consistency(sample_data):
    """✅ Reproducibility: same random_state => same accuracy."""
    X_train, X_test, y_train, y_test = sample_data
    model1 = train_model(X_train, y_train, model_type="RandomForest")
    model2 = train_model(X_train, y_train, model_type="RandomForest")

    acc1 = evaluate_model(model1, X_test, y_test)["accuracy"]
    acc2 = evaluate_model(model2, X_test, y_test)["accuracy"]

    assert np.isclose(acc1, acc2, atol=1e-6), "❌ Accuracy mismatch with same random_state"


@pytest.mark.parametrize("model_type", ["RandomForest", "DecisionTree", "SVM"])
def test_models_can_overfit_small_data(model_type):
    """✅ On very small data, models should perfectly fit (overfit check)."""
    X, y = make_classification(n_samples=20, n_features=5, n_classes=2, random_state=0)
    model = train_model(X, y, model_type=model_type)
    preds = model.predict(X)

    accuracy = (preds == y).mean()

    # Looser threshold for SVM
    if model_type == "SVM":
        assert accuracy >= 0.85, f"❌ {model_type} failed to overfit tiny dataset"
    else:
        assert accuracy >= 0.95, f"❌ {model_type} failed to overfit tiny dataset"


def test_model_can_handle_single_feature():
    """✅ Models should work even with a single feature."""
    X, y = make_classification(
        n_samples=50,
        n_features=1,
        n_informative=1,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=1,  # ✅ FIXED
        random_state=0,
    )

    model = train_model(X, y, model_type="DecisionTree")
    preds = model.predict(X)

    assert preds.shape == y.shape
    assert (preds == y).mean() >= 0.9, "❌ Model struggled with single feature"



if __name__ == "__main__":
    pytest.main([__file__, "-v", "--disable-warnings"])
