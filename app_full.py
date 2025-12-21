import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import onnxruntime as ort
import pickle

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

# ===== ZAŁADUJ MODEL ONNX =====
@st.cache_resource
def load_onnx_model():
    """Załaduj ONNX model"""
    model_path = r"C:\Users\slast\PYTHON\0_projekty do portfolio\07_pytorch_cl\model.onnx"
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    return sess

@st.cache_resource
def load_scaler_params():
    """Załaduj parametry skalowania"""
    scaler_path = r"C:\Users\slast\PYTHON\0_projekty do portfolio\07_pytorch_cl\data\procced\scaler_params.pkl"
    try:
        with open(scaler_path, 'rb') as f:
            params = pickle.load(f)
        return params
    except:
        return None

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
            from rdkit.Chem import Descriptors, Crippen, Lipinski
            
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
            # Fallback: użyj średnich wartości
            numeric_features = np.zeros(6, dtype=np.float32)
        
        # Połącz Morgan FP + numeric features
        features = np.concatenate([morgan_features, numeric_features])
        
        return features, "✅ OK"
    
    except Exception as e:
        return None, f"❌ Błąd: {str(e)}"

def predict_logp_onnx(features, sess):
    """
    Predykcja logP używając ONNX modelu
    """
    try:
        # Reshape dla batch
        X = features.reshape(1, -1).astype(np.float32)
        
        # Inference
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        
        result = sess.run([output_name], {input_name: X})
        logp = float(result[0][0][0])
        
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
    sess = load_onnx_model()
    scaler_params = load_scaler_params()

st.success('✅ Model załadowany (ONNX)!')

# ===== SIDEBAR - INFORMACJE =====
with st.sidebar:
    st.title("ℹ️ Informacje")
    
    st.markdown("""
    **O aplikacji:**
    - Model: Neural Network (ONNX)
    - Features: Morgan Fingerprints (2048 bitów) + 6 deskryptorów
    - Target: logP (lipofilowość)
    - Dokładność: R² = 0.9007 na test set
    - Framework: PyTorch (konwertowany do ONNX)
    
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
    st.write(f"- Format: ONNX (4.66 MB)")

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
            # Predykcja ONNX
            logp = predict_logp_onnx(features, sess)
            
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
    st.caption("Neural Network (ONNX)")
    st.caption("Morgan Fingerprints 2048")

with col2:
    st.markdown("**📈 Wydajność**")
    st.caption("Test R²: 0.9007")
    st.caption("MAE: 0.6166")

with col3:
    st.markdown("**🔗 Technologia**")
    st.caption("ONNX Runtime")
    st.caption("RDKit | Streamlit")