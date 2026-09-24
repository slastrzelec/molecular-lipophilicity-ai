import os

import numpy as np
import pytest

import logp_utils

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
ETHANOL = "CCO"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.fixture(scope="module")
def scaler_params():
    return logp_utils.load_scaler_params(base_dir=REPO_ROOT)


@pytest.fixture(scope="module")
def model():
    return logp_utils.load_model(base_dir=REPO_ROOT)


# ---- smiles_to_features ----

def test_valid_smiles_returns_correct_feature_shape(scaler_params):
    features, status = logp_utils.smiles_to_features(ASPIRIN, scaler_params)
    assert features is not None
    assert features.shape == (logp_utils.MORGAN_NBITS + 6,)
    assert status.startswith("✅")


def test_invalid_smiles_returns_none_and_error():
    features, status = logp_utils.smiles_to_features("not a smiles!!", None)
    assert features is None
    assert status.startswith("❌")


def test_empty_smiles_returns_none_and_error():
    features, status = logp_utils.smiles_to_features("", None)
    assert features is None
    assert status.startswith("❌")


def test_valid_smiles_numeric_descriptors_are_not_all_zero(scaler_params):
    """Regression test for the fixed silent-except bug: a valid molecule must
    produce real (non-fallback) physicochemical descriptors, not six zeros."""
    features, _ = logp_utils.smiles_to_features(ASPIRIN, scaler_params)
    numeric_part = features[logp_utils.MORGAN_NBITS:]
    assert not np.allclose(numeric_part, 0.0)


def test_descriptor_failure_is_surfaced_not_silenced():
    """With malformed scaler_params (missing expected keys), feature scaling
    must raise and be reported as an error -- not silently produce zeros."""
    bad_scaler_params = {"feature_cols": ["mol_weight"], "mean": {}, "scale": {}}
    features, status = logp_utils.smiles_to_features(ASPIRIN, bad_scaler_params)
    assert features is None
    assert status.startswith("❌")


# ---- predict_logp ----

def test_predict_logp_returns_finite_float(model, scaler_params):
    features, _ = logp_utils.smiles_to_features(ASPIRIN, scaler_params)
    logp = logp_utils.predict_logp(features, model)
    assert isinstance(logp, float)
    assert np.isfinite(logp)
    # Sanity range check, not an exact-value assertion (model is stochastic-free
    # at eval time but this guards against a badly broken pipeline/checkpoint).
    assert -15 < logp < 15


def test_different_substrates_give_different_predictions(model, scaler_params):
    """Regression test: distinct molecules must yield distinct predictions
    (guards against accidentally predicting on a constant/placeholder input)."""
    feat_a, _ = logp_utils.smiles_to_features(ETHANOL, scaler_params)
    feat_b, _ = logp_utils.smiles_to_features(CAFFEINE, scaler_params)
    logp_a = logp_utils.predict_logp(feat_a, model)
    logp_b = logp_utils.predict_logp(feat_b, model)
    assert logp_a != pytest.approx(logp_b, abs=1e-6)


# ---- interpret_logp ----

@pytest.mark.parametrize(
    "logp,expected_category",
    [
        (-5, "hydrophilic"),
        (-2.5, "hydrophilic"),
        (-1, "hydrophilic"),
        (1, "moderate"),
        (3, "lipophilic"),
        (7, "highly_lipophilic"),
    ],
)
def test_interpret_logp_categories(logp, expected_category):
    _, _, category = logp_utils.interpret_logp(logp)
    assert category == expected_category


def test_interpret_logp_boundaries_are_left_closed():
    # logp < -2 -> hydrophilic; logp == -2 -> next bucket (hydrophilic still, per <0)
    _, _, cat_below = logp_utils.interpret_logp(-2.0001)
    _, _, cat_at = logp_utils.interpret_logp(-2.0)
    assert cat_below == "hydrophilic"
    assert cat_at == "hydrophilic"
