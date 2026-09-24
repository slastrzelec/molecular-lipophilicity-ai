import streamlit as st

import logp_utils

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
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    
    .result-box {
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid;
        background: rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
        margin: 15px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .success-message {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        color: #1a5f3d;
        padding: 15px 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ===== LOAD MODEL & CHEMISTRY LOGIC (logp_utils.py) =====
@st.cache_resource
def load_model():
    return logp_utils.load_model()


@st.cache_resource
def load_scaler_params():
    return logp_utils.load_scaler_params()


smiles_to_features = logp_utils.smiles_to_features
predict_logp = logp_utils.predict_logp
interpret_logp = logp_utils.interpret_logp

# ===== HEADER =====
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="main-title">🧬 logP Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Predict molecular lipophilicity from SMILES structure</p>', unsafe_allow_html=True)

# Load model
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
    
    smiles_input = st.text_input(
        "Enter SMILES:",
        value="CCO",
        placeholder="e.g., CCO for ethanol",
        label_visibility="collapsed"
    )
    
    st.caption("💡 Generate SMILES at: [PubChem](https://pubchem.ncbi.nlm.nih.gov/)")
    
    # Draw molecule using PubChem
    if smiles_input:
        try:
            st.image(
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles_input}/PNG",
                caption="Molecular Structure",
                width=300
            )
        except:
            st.warning("⚠️ Could not visualize structure")

with col2:
    st.subheader("🔬 Prediction Result")
    
    if smiles_input:
        features, status = smiles_to_features(smiles_input, scaler_params)
        
        st.write(f"**Status**: {status}")
        
        if features is not None:
            logp = predict_logp(features, model)
            
            if logp is not None:
                interpretation, color, category = interpret_logp(logp)
                
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