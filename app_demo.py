import streamlit as st
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

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

# ===== FUNKCJE =====
MORGAN_RADIUS = 2
MORGAN_NBITS = 2048

def smiles_to_morgan_fp(smiles):
    """
    Konwertuj SMILES na Morgan fingerprint
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "❌ Niepoprawny SMILES"
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_NBITS)
        return np.array(fp, dtype=np.float32), "✅ OK"
    
    except Exception as e:
        return None, f"❌ Błąd: {str(e)}"

def simple_logp_prediction(smiles):
    """
    Prosta predykcja logP (demo - bez ML modelu)
    Opiera się na liczbie atomów i strukturze
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Prosta heurystyka
        num_atoms = mol.GetNumAtoms()
        num_rotatable = sum(1 for bond in mol.GetBonds() if bond.GetIsRotatable())
        num_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        num_donors = sum(1 for atom in mol.GetAtoms() if atom.GetTotalDegree() == 1 and atom.GetSymbol() in ['N', 'O'])
        
        # Empiryczna formuła
        logp = (num_atoms * 0.1 + 
                num_rotatable * 0.3 + 
                num_aromatic * 0.5 - 
                num_donors * 0.4 - 
                2.0)
        
        return float(logp)
    
    except:
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
st.markdown('<h1 class="main-title">🧬 logP Predictor (DEMO)</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predykcja lipofilowości cząsteczek na podstawie struktury SMILES</p>', 
            unsafe_allow_html=True)

st.warning("⚠️ **DEMO VERSION** - Używa uproszczonego algorytmu bez ML modelu. Pełna wersja wymaga PyTorch.")

# ===== SIDEBAR - INFORMACJE =====
with st.sidebar:
    st.title("ℹ️ Informacje")
    
    st.markdown("""
    **O aplikacji:**
    - Version: DEMO (bez ML)
    - Features: Morgan Fingerprints analiza
    - Target: logP (lipofilowość)
    - Status: Uproszczona heurystyka
    
    **Co to logP?**
    - logP = log(P(octanol)/P(water))
    - Mierzy rozpuszczalność w tłuszczach vs wodzie
    - logP < 0: hydrofilne (rozpuszczalne w wodzie)
    - logP > 0: lipidowe (rozpuszczalne w tłuszczach)
    
    **Format SMILES:**
    - CC(=O)O - octan (aspiryna)
    - CCO - etanol
    - c1ccccc1 - benzen
    
    **Pełna wersja:**
    - Neural Network (PyTorch)
    - Test R²: 0.9007
    """)
    
    st.divider()
    st.info("💡 Ta wersja DEMO pokazuje interfejs. Pełny model będzie deployowany na Streamlit Cloud.")

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
        fp, status = smiles_to_morgan_fp(smiles_input)
        
        st.write(f"Status: {status}")
        
        if fp is not None:
            # Predykcja
            logp = simple_logp_prediction(smiles_input)
            
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

# ===== INFO O PEŁNEJ WERSJI =====
st.divider()
st.markdown("""
### 🚀 Pełna wersja (z ML modelem)

Ta aplikacja DEMO używa uproszczonej heurystyki. 

**Pełna wersja będzie mieć:**
- ✅ Neural Network wytrenowany na 11,612 cząsteczkach
- ✅ Morgan Fingerprints (2048 bitów)
- ✅ Test R² = 0.9007
- ✅ RMSE = 0.9332
- ✅ Hostowana na Streamlit Community Cloud

**Status:** Rozwiązuję problem z PyTorch, aplikacja wkrótce!
""")

# ===== FOOTER =====
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Model Info**")
    st.caption("Neural Network (PyTorch)")
    st.caption("Morgan Fingerprints 2048")

with col2:
    st.markdown("**📈 Wydajność (pełna wersja)**")
    st.caption("Test R²: 0.9007")
    st.caption("MAE: 0.6166")

with col3:
    st.markdown("**🔗 Status**")
    st.caption("⚠️ DEMO version")
    st.caption("🚀 Full version coming soon")