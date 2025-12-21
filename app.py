import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import pickle
import os

# ===== KONFIGURACJA STREAMLIT =====
st.set_page_config(
    page_title="logP Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1em;
        margin-bottom: 30px;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===== ZAŁADUJ MODEL =====
@st.cache_resource
def load_model():
    """Załaduj wytrenowany model PyTorch"""
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
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "checkpoints", "best_model.pt")
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model

@st.cache_resource
def load_scaler_params():
    """Załaduj parametry skalowania"""
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    scaler_path = os.path.join(current_dir, "data", "procced", "scaler_params.pkl")
    with open(scaler_path, 'rb') as f:
        params = pickle.load(f)
    return params

# ===== FUNKCJE =====
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

def smiles_to_features(smiles, scaler_params):
    """
    Konwertuj SMILES na cechy (Morgan FP + numeric)
    """
    try:
        # Morgan fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "❌ Niepoprawny SMILES"
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        morgan_features = np.array(fp, dtype=np.float32)
        
        # Oblicz numeric features z cząsteczki
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
            
            # Skaluj numeric features
            if scaler_params is not None:
                feature_cols = scaler_params['feature_cols']
                for i, col in enumerate(feature_cols):
                    mean = scaler_params['mean'][col]
                    scale = scaler_params['scale'][col]
                    numeric_features_raw[i] = (numeric_features_raw[i] - mean) / scale
            
            numeric_features = numeric_features_raw
            
        except:
            # Fallback: użyj zer
            numeric_features = np.zeros(6, dtype=np.float32)
        
        # Połącz Morgan FP + numeric features
        features = np.concatenate([morgan_features, numeric_features])
        
        return features, "✅ OK"
    
    except Exception as e:
        return None, f"❌ Błąd: {str(e)}"

def predict_logp(features, model):
    """
    Predykcja logP za pomocą PyTorch modelu
    """
    try:
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0)
            logp = model(X).item()
        return logp
    except Exception as e:
        return None

def interpret_logp(logp):
    """
    Interpretacja wartości logP
    """
    if logp < -2:
        return "🔵 Bardzo hydrofilne (rozpuszczalne w wodzie)", "#0099ff"
    elif logp < 0:
        return "🟢 Hydrofilne", "#00cc66"
    elif logp < 2:
        return "🟡 Umiarkowana lipidowość", "#ffcc00"
    elif logp < 5:
        return "🟠 Lipidowe", "#ff9900"
    else:
        return "🔴 Bardzo lipidowe (rozpuszczalne w tłuszczach)", "#ff3333"

# ===== UI =====
st.markdown('<h1 class="main-title">🧬 logP Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predykcja lipofilowości cząsteczek na podstawie struktury SMILES</p>', 
            unsafe_allow_html=True)

# Załaduj model i scaler
with st.spinner('⏳ Ładowanie modelu...'):
    try:
        model = load_model()
        scaler_params = load_scaler_params()
        st.success('✅ Model załadowany (PyTorch)!')
    except Exception as e:
        st.error(f"❌ Błąd ładowania modelu: {str(e)}")
        st.stop()

# ===== SIDEBAR - INFORMACJE =====
with st.sidebar:
    st.title("ℹ️ Informacje")
    
    st.markdown("""
    **O aplikacji:**
    - Model: Neural Network (PyTorch)
    - Features: Morgan Fingerprints (2048 bitów) + 6 deskryptorów
    - Target: logP (lipofilowość)
    - Dokładność: R² = 0.9007 na test set
    
    **Co to logP?**
    - logP = log(P(octanol)/P(water))
    - Mierzy rozpuszczalność w tłuszczach vs wodzie
    - logP < 0: hydrofilne (rozpuszczalne w wodzie)
    - logP > 0: lipidowe (rozpuszczalne w tłuszczach)
    
    **Format SMILES:**
    - CC(=O)O - octan (aspiryna)
    - CCO - etanol
    - c1ccccc1 - benzen
    """)
    
    st.divider()
    st.markdown("**Model Info:**")
    st.write(f"- Parametry: 1,218,305")
    st.write(f"- Train R²: 0.9948")
    st.write(f"- Val R²: 0.9093")
    st.write(f"- Test R²: 0.9007")

# ===== MAIN CONTENT =====
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Wpisz SMILES")
    
    # Przykłady
    examples = {
        "Octan (Aspiryna)": "CC(=O)OC(CC(=O)O)C[N+](C)(C)C",
        "Etanol": "CCO",
        "Benzen": "c1ccccc1",
        "Kofaina": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Toluene": "Cc1ccccc1",
        "Aceton": "CC(=O)C",
        "Fenol": "Oc1ccccc1",
        "Naftalin": "c1cc2ccccc2cc1",
    }
    
    selected_example = st.selectbox(
        "Lub wybierz przykład:",
        ["---"] + list(examples.keys())
    )
    
    if selected_example != "---":
        smiles_input = examples[selected_example]
    else:
        smiles_input = st.text_input(
            "Wpisz SMILES cząsteczki:",
            placeholder="np. CC(=O)O",
            label_visibility="collapsed"
        )
    
    st.info("💡 Możesz wygenerować SMILES na: https://pubchem.ncbi.nlm.nih.gov/")

with col2:
    st.subheader("🔬 Wynik Predykcji")
    
    if smiles_input:
        # Przetwórz SMILES
        features, status = smiles_to_features(smiles_input, scaler_params)
        
        st.write(f"Status: {status}")
        
        if features is not None:
            # Predykcja PyTorch
            logp = predict_logp(features, model)
            
            if logp is not None:
                # Interpretacja
                interpretation, color = interpret_logp(logp)
                
                # Wyświetl wynik
                st.markdown(f"""
                <div class="metric-box" style="border-left: 5px solid {color};">
                    <h2 style="color: {color}; margin: 0;">logP = {logp:.2f}</h2>
                    <p style="font-size: 1.1em; margin: 10px 0 0 0;">{interpretation}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Dodatkowe info
                st.markdown("---")
                st.write("**Szczegółowa interpretacja:**")
                
                if logp < -2:
                    st.write("""
                    🔵 **Bardzo hydrofilne (rozpuszczalne w wodzie)**
                    - Dobrze rozpuszcza się w wodzie
                    - Słabo przechodzi przez błonę biologiczną
                    - Słaba absorpcja żołądkowo-jelitowa
                    """)
                elif logp < 0:
                    st.write("""
                    🟢 **Hydrofilne**
                    - Dobrze rozpuszcza się w wodzie
                    - Umiarkowana permeabilność błonowa
                    - Może mieć dobrą bioprzedostępność
                    """)
                elif logp < 2:
                    st.write("""
                    🟡 **Umiarkowana lipofilowość**
                    - Optimalny balans hydrofobowości
                    - Najczęściej pożądane dla leków
                    - Dobra bioprzedostępność
                    """)
                elif logp < 5:
                    st.write("""
                    🟠 **Lipidowe**
                    - Dobrze rozpuszcza się w tłuszczach
                    - Wysoka permeabilność błonowa
                    - Ryzyko akumulacji w tkankach tłuszczowych
                    """)
                else:
                    st.write("""
                    🔴 **Bardzo lipidowe (rozpuszczalne w tłuszczach)**
                    - Słabo rozpuszcza się w wodzie
                    - Wysoka permeabilność błonowa
                    - Duże ryzyko toksyczności i efektów ubocznych
                    """)
            else:
                st.error("⚠️ Nie mogę obliczyć logP")
        else:
            st.error(f"⚠️ Błąd przetwarzania: {status}")
    else:
        st.info("👆 Wpisz SMILES w lewej kolumnie, aby zobaczyć predykcję")

# ===== FOOTER =====
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Model Info**")
    st.caption("PyTorch Neural Network")
    st.caption("Morgan Fingerprints 2048")

with col2:
    st.markdown("**📈 Wydajność**")
    st.caption("Test R²: 0.9007")
    st.caption("MAE: 0.6166")

with col3:
    st.markdown("**🔗 Linki**")
    st.caption("[GitHub](https://github.com)")
    st.caption("[PubChem](https://pubchem.ncbi.nlm.nih.gov/)")