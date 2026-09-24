"""Chemistry and model logic for the logP Predictor, decoupled from Streamlit.

Kept separate from app.py so it can be unit-tested without a Streamlit
runtime context (importing app.py directly would execute st.set_page_config
and the rest of the UI at module load time).
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

MORGAN_RADIUS = 2
MORGAN_NBITS = 2048


class MoleculeLogPPredictor(nn.Module):
    """4-layer MLP: 2054 -> 512 -> 256 -> 128 -> 1, BatchNorm + ReLU + Dropout(0.3)."""

    def __init__(self, input_size=2054, dropout_rate=0.3):
        super(MoleculeLogPPredictor, self).__init__()

        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)

        self.fc4 = nn.Linear(128, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        return x


def load_model(base_dir=None):
    """Load the trained PyTorch model from checkpoints/best_model.pt."""
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "checkpoints", "best_model.pt")

    model = MoleculeLogPPredictor(input_size=2054)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def load_scaler_params(base_dir=None):
    """Load the scaler parameters used to normalize the 6 numeric descriptors."""
    base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(base_dir, "data", "procced", "scaler_params.pkl")
    with open(scaler_path, "rb") as f:
        return pickle.load(f)


def smiles_to_features(smiles, scaler_params):
    """Convert a SMILES string to a (2054,) feature vector: Morgan FP + 6 scaled descriptors.

    Returns (features, status_message). features is None on failure.

    Note: the 6 physicochemical descriptors are computed with a plain,
    unguarded call to RDKit's Descriptors module. An earlier version of
    this function wrapped that block in a bare `except: numeric_features =
    zeros(6)`, which silently fed the model six zeroed-out features instead
    of surfacing the failure -- the same "hidden except swallows a real
    bug" pattern found (and fixed) in the Buchwald-Hartwig project. Any
    descriptor-computation failure now propagates to the outer except below,
    which returns a visible error instead of a silently degraded prediction.
    """
    try:
        if not smiles or not smiles.strip():
            return None, "❌ Empty SMILES"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return None, "❌ Invalid SMILES format"

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        morgan_features = np.array(fp, dtype=np.float32)

        mol_weight = Descriptors.MolWt(mol)
        polar_area = Descriptors.TPSA(mol)
        complexity = Descriptors.BertzCT(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)

        numeric_features = np.array(
            [mol_weight, polar_area, complexity, h_donors, h_acceptors, rotatable_bonds],
            dtype=np.float32,
        )

        if scaler_params is not None:
            feature_cols = scaler_params["feature_cols"]
            for i, col in enumerate(feature_cols):
                mean = scaler_params["mean"][col]
                scale = scaler_params["scale"][col]
                numeric_features[i] = (numeric_features[i] - mean) / scale

        features = np.concatenate([morgan_features, numeric_features])
        return features, "✅ Processing successful"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def predict_logp(features, model):
    """Run the model on a (2054,) feature vector and return a scalar logP prediction."""
    try:
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0)
            logp = model(X).item()
        return logp
    except Exception:
        return None


def interpret_logp(logp):
    """Map a logP value to (label, color, category)."""
    if logp < -2:
        return "🔵 Highly Hydrophilic (Water-soluble)", "#0099ff", "hydrophilic"
    elif logp < 0:
        return "🟢 Hydrophilic", "#00cc66", "hydrophilic"
    elif logp < 2:
        return "🟡 Moderate Lipophilicity (OPTIMAL)", "#ffcc00", "moderate"
    elif logp < 5:
        return "🟠 Lipophilic", "#ff9900", "lipophilic"
    else:
        return "🔴 Highly Lipophilic (Fat-soluble)", "#ff3333", "highly_lipophilic"
