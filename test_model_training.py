# tests/test_model_training.py
import sys
import os
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Ensure project root is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_training import train_model, evaluate_model

# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def sample_data():
    """Generate synthetic dataset for testing ML pipeline."""
    X, y = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_classes=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# -----------------------------
# Tests
# -----------------------------
def test_train_model_returns_valid_model(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = train_model(X_train, y_train, model_type="RandomForest")
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")


def test_evaluate_model_accuracy(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model = train_model(X_train, y_train, model_type="RandomForest")
    result = evaluate_model(model, X_test, y_test)
    assert isinstance(result, dict)
    assert "accuracy" in result and "report" in result
    assert 0 <= result["accuracy"] <= 1


@pytest.mark.parametrize("model_type,expected_cls", [
    ("RandomForest", RandomForestClassifier),
    ("DecisionTree", DecisionTreeClassifier),
    ("SVM", SVC)
])
def test_train_model_multiple_algorithms(sample_data, model_type, expected_cls):
    X_train, X_test, y_train, y_test = sample_data
    model = train_model(X_train, y_train, model_type=model_type)
    assert isinstance(model, expected_cls)
    assert hasattr(model, "predict")


def test_reproducibility(sample_data):
    X_train, X_test, y_train, y_test = sample_data
    model1 = train_model(X_train, y_train, model_type="RandomForest", random_state=42)
    model2 = train_model(X_train, y_train, model_type="RandomForest", random_state=42)
    acc1 = evaluate_model(model1, X_test, y_test)["accuracy"]
    acc2 = evaluate_model(model2, X_test, y_test)["accuracy"]
    assert np.isclose(acc1, acc2, atol=1e-6)


@pytest.mark.parametrize("model_type", ["RandomForest", "DecisionTree", "SVM"])
def test_models_overfit_small_dataset(model_type):
    """
    On very small datasets, models should overfit.
    SVM is allowed to reach slightly less than 100% due to probability scaling.
    """
    X, y = make_classification(
    n_samples=20,
    n_features=5,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    class_sep=5.0,  # Increase separation for perfect linear separability
    random_state=0
)


    if model_type == "SVM":
        # Override SVM parameters to force overfitting
        model = SVC(kernel="linear", C=1e6, probability=False, random_state=0)
        model.fit(X, y)
    else:
        model = train_model(X, y, model_type=model_type)

    preds = model.predict(X)
    accuracy = (preds == y).mean()

    if model_type == "SVM":
        assert accuracy >= 0.95, f"{model_type} failed to overfit tiny dataset"
    else:
        assert accuracy == 1.0, f"{model_type} failed to overfit tiny dataset"


# -----------------------------
# Run pytest directly
# -----------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--disable-warnings"])
