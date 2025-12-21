import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import os

# ===== STREAMLIT CONFIGURATION =====
st.set_page_config(
    page_title="logP Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CUSTOM CSS STYLING =====
st.markdown("""
    <style>
    /* Main title */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 40px 20px;
        border-radius: 15px;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    /* Result box */
    .result-box {
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Success message */
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1a5f3d;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 2px solid #ddd;
    }
    
    /* Sidebar styling */
    .sidebar-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===== LOAD MODEL =====
@st.cache_resource
def load_model():
    """Load trained PyTorch model"""
    class MoleculeLogPPredictor(nn.Module):
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
    
    model = MoleculeLogPPredictor(input_size=2054)
    
    # Relative path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "checkpoints", "best_model.pt")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model

@st.cache_resource
def load_scaler_params():
    """Load scaling parameters"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(current_dir, "data", "procced", "scaler_params.pkl")
    with open(scaler_path, 'rb') as f:
        params = pickle.load(f)
    return params

# ===== FUNCTIONS =====
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

def smiles_to_features(smiles, scaler_params):
    """Convert SMILES to features (Morgan FP + numeric)"""
    try:
        # Morgan fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "❌ Invalid SMILES format"
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        morgan_features = np.array(fp, dtype=np.float32)
        
        # Calculate numeric features
        try:
            from rdkit.Chem import Descriptors
            
            mol_weight = Descriptors.MolWt(mol)
            polar_area = Descriptors.TPSA(mol)
            complexity = Descriptors.BertzCT(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            numeric_features_raw = np.array([
                mol_weight,
                polar_area,
                complexity,
                h_donors,
                h_acceptors,
                rotatable_bonds
            ], dtype=np.float32)
            
            # Scale numeric features
            if scaler_params is not None:
                feature_cols = scaler_params['feature_cols']
                for i, col in enumerate(feature_cols):
                    mean = scaler_params['mean'][col]
                    scale = scaler_params['scale'][col]
                    numeric_features_raw[i] = (numeric_features_raw[i] - mean) / scale
            
            numeric_features = numeric_features_raw
            
        except:
            numeric_features = np.zeros(6, dtype=np.float32)
        
        # Combine Morgan FP + numeric features
        features = np.concatenate([morgan_features, numeric_features])
        
        return features, "✅ Processing successful"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

def predict_logp(features, model):
    """Predict logP using PyTorch model"""
    try:
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0)
            logp = model(X).item()
        return logp
    except Exception as e:
        return None

def interpret_logp(logp):
    """Interpret logP value"""
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

# ===== HEADER =====
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-title">🧬 logP Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict molecular lipophilicity from SMILES structure</p>', unsafe_allow_html=True)

# Load model and scaler
with st.spinner('⏳ Loading model...'):
    try:
        model = load_model()
        scaler_params = load_scaler_params()
        st.markdown('<div class="success-message">✅ Model loaded successfully (PyTorch)</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"❌ Model loading error: {str(e)}")
        st.stop()

# ===== SIDEBAR =====
with st.sidebar:
    st.markdown("## 📊 Model Information")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test R²", "0.9007")
        st.metric("Train R²", "0.9948")
    with col2:
        st.metric("MAE", "0.6166")
        st.metric("RMSE", "0.9332")
    
    st.divider()
    
    st.markdown("## ℹ️ About logP")
    st.info("""
    **logP** = log(Partition Coefficient)
    
    Measures molecule distribution between octanol and water.
    
    - **logP < 0**: Water-loving (hydrophilic)
    - **0 < logP < 2**: Optimal for drugs
    - **logP > 2**: Fat-loving (lipophilic)
    """)
    
    st.divider()
    
    st.markdown("## 🛠️ Technical Details")
    st.write("""
    - **Model**: Neural Network (PyTorch)
    - **Features**: Morgan FP (2048) + 6 descriptors
    - **Parameters**: 1,218,305
    - **Training Data**: 11,612 molecules
    """)

# ===== MAIN CONTENT =====
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📝 Input SMILES")
    
    examples = {
        "🔬 Aspirin": "CC(=O)OC(CC(=O)O)C[N+](C)(C)C",
        "🍸 Ethanol": "CCO",
        "⚛️ Benzene": "c1ccccc1",
        "☕ Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "🌲 Toluene": "Cc1ccccc1",
        "🧪 Acetone": "CC(=O)C",
        "🌺 Phenol": "Oc1ccccc1",
        "💎 Naphthalene": "c1cc2ccccc2cc1",
    }
    
    selected_example = st.selectbox(
        "Select example or enter custom SMILES:",
        ["Custom"] + list(examples.keys())
    )
    
    if selected_example == "Custom":
        smiles_input = st.text_input(
            "Enter SMILES:",
            placeholder="e.g., CCO for ethanol",
            label_visibility="collapsed"
        )
    else:
        smiles_input = examples[selected_example]
    
    st.caption("💡 Generate SMILES at: [PubChem](https://pubchem.ncbi.nlm.nih.gov/)")

with col2:
    st.subheader("🔬 Prediction Result")
    
    if smiles_input:
        # Process SMILES
        features, status = smiles_to_features(smiles_input, scaler_params)
        
        st.write(f"**Status**: {status}")
        
        if features is not None:
            # Predict
            logp = predict_logp(features, model)
            
            if logp is not None:
                # Interpret
                interpretation, color, category = interpret_logp(logp)
                
                # Display result
                st.markdown(f"""
                <div class="result-box" style="border-color: {color};">
                    <div style="font-size: 2.5em; color: {color}; font-weight: bold; margin-bottom: 10px;">
                        logP = {logp:.2f}
                    </div>
                    <div style="font-size: 1.1em; color: {color};">
                        {interpretation}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed explanation
                st.divider()
                st.markdown("### 📖 Detailed Interpretation")
                
                explanations = {
                    "hydrophilic": """
                    **Highly Water-Soluble**
                    - ✅ Good water solubility
                    - ✅ Poor membrane permeability
                    - ❌ Low oral absorption
                    - Use: Hydrophilic drugs, diagnostic agents
                    """,
                    "moderate": """
                    **Optimal Balance** ⭐
                    - ✅ Good water solubility
                    - ✅ Good membrane permeability
                    - ✅ High oral bioavailability
                    - Use: Most commercial drugs
                    """,
                    "lipophilic": """
                    **Fat-Soluble**
                    - ✅ Good membrane permeability
                    - ❌ Poor water solubility
                    - ⚠️ Tissue accumulation risk
                    - Use: Lipid-targeting drugs
                    """
                }
                
                st.write(explanations.get(category, ""))
            else:
                st.error("❌ Could not calculate logP")
        else:
            st.error(f"⚠️ {status}")
    else:
        st.info("👈 Enter or select SMILES to see prediction")

# ===== FOOTER =====
st.divider()

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("### 📊 Architecture")
    st.write("Input (2054) → 512 → 256 → 128 → Output (1)")

with footer_col2:
    st.markdown("### 🎯 Performance")
    st.write("**Test R²**: 0.9007  \n**MAE**: 0.6166")

with footer_col3:
    st.markdown("### 🔗 Links")
    st.write("[GitHub](https://github.com/slastrzelec/molecular-lipophilicity-ai) | [PubChem](https://pubchem.ncbi.nlm.nih.gov/)")