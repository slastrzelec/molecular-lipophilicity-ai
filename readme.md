# 🧬 logP Predictor

Deep Learning application for predicting molecular lipophilicity (logP) based on SMILES structure, using a neural network model trained with PyTorch.

## 📋 About the Project

**logP (Partition Coefficient)** is the logarithm of the distribution coefficient of a molecule between octanol and water. It's a key descriptor in chemistry, pharmacy, and biotechnology, determining:

- Water vs fat solubility
- Biological membrane permeability
- Drug bioavailability
- Molecular hydrophobicity

## 🎯 Model Characteristics

| Parameter | Value |
|-----------|-------|
| **Model** | Neural Network (PyTorch) |
| **Features** | Morgan Fingerprints (2048 bits) + 6 molecular descriptors |
| **Training Data** | 11,612 molecules from PubChem |
| **Test R²** | 0.9007 |
| **MAE** | 0.6166 |
| **RMSE** | 0.9332 |
| **Model Parameters** | 1,218,305 |

## 📊 Model Architecture

```
Input (2054)
    ↓
FC1 (512) → BatchNorm → ReLU → Dropout(0.3)
    ↓
FC2 (256) → BatchNorm → ReLU → Dropout(0.3)
    ↓
FC3 (128) → BatchNorm → ReLU → Dropout(0.3)
    ↓
FC4 (1) → Output (logP)
```

## 🚀 Installation

### Requirements
- Python 3.8+
- Conda (optional)

### Installation Steps

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/logp-predictor.git
cd logp-predictor
```

2. **Create environment (optional):**
```bash
conda create -n logp-env python=3.10
conda activate logp-env
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## 🏃 Running the Application

### Locally
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### On Streamlit Cloud
```bash
streamlit run app.py --server.port 8501
```

## 📁 Project Structure

```
logp-predictor/
├── app.py                          # Streamlit application
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── checkpoints/
│   └── best_model.pt              # Trained PyTorch model
├── data/
│   ├── raw/
│   │   └── RAW.csv                # Raw data
│   └── procced/
│       ├── data_preprocessed.csv  # Preprocessed data
│       └── scaler_params.pkl      # Scaling parameters
├── visualisation/                  # Plots and visualizations
│   ├── 01_xlogp_distribution.png
│   ├── 02_xlogp_outliers_analysis.png
│   ├── 03_xlogp_before_after_outliers.png
│   ├── 04_features_scaling_comparison.png
│   ├── 05_morgan_fingerprints_analysis.png
│   ├── 06_data_split_distribution.png
│   └── 07_training_results.png
└── notebooks/
    ├── EDA.ipynb                  # Exploratory Data Analysis
    └── pytorch_logP_pred.ipynb    # Model training
```

## 🔬 How to Use the Application

1. **Choose or enter SMILES:**
   - Enter SMILES manually
   - Or select from examples (aspirin, ethanol, benzene, etc.)
   - Generate SMILES at: https://pubchem.ncbi.nlm.nih.gov/

2. **The application calculates:**
   - Morgan Fingerprint (2048 bits)
   - Molecular descriptors (Molecular Weight, TPSA, etc.)
   - logP prediction using the model

3. **You will see:**
   - logP value
   - Color-coded interpretation (🔵🟢🟡🟠🔴)
   - Detailed explanation

## 📈 Result Interpretation

| logP | Interpretation | Color |
|------|----------------|-------|
| < -2 | Very hydrophilic | 🔵 |
| -2 to 0 | Hydrophilic | 🟢 |
| 0 to 2 | Moderate lipophilicity (OPTIMAL) | 🟡 |
| 2 to 5 | Lipophilic | 🟠 |
| > 5 | Highly lipophilic | 🔴 |

## 🔧 Technology Stack

### Backend
- **PyTorch** - Deep Learning framework
- **RDKit** - Cheminformatics library
- **NumPy/Pandas** - Data processing
- **scikit-learn** - Preprocessing

### Frontend
- **Streamlit** - Web application framework
- **Matplotlib/Seaborn** - Visualizations

### Cloud
- **Streamlit Community Cloud** - Hosting

## 📊 Training the Model

To train the model from scratch:

```bash
jupyter notebook pytorch_logP_pred.ipynb
```

The notebook includes:
- Exploratory Data Analysis (EDA)
- Feature engineering (Morgan Fingerprints)
- Model training
- Evaluation and visualizations
- Hyperparameter tuning

## 📥 Dataset

Data sourced from **PubChem** (https://pubchem.ncbi.nlm.nih.gov/)

- **11,612 molecules** with logP values
- **Features:** Molecular Weight, Polar Area, Complexity, H-Bond Donors/Acceptors, Rotatable Bonds
- **Preprocessing:** StandardScaler normalization, outlier removal

## 🎓 Training Results

```
Training Duration: 8m 51s
Epochs: 72/100 (Early Stopping)

Metrics:
├── Train R²: 0.9948
├── Val R²: 0.9093
└── Test R²: 0.9007

Errors:
├── Train MAE: 0.1411
├── Val MAE: 0.6045
└── Test MAE: 0.6166
```

## 🚀 Deployment

### Streamlit Cloud

1. **Push to GitHub:**
```bash
git add .
git commit -m "Deploy logP predictor"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to https://streamlit.io/cloud
   - Log in with GitHub
   - Select repository
   - Deploy!

### Docker (optional)

```bash
docker build -t logp-predictor .
docker run -p 8501:8501 logp-predictor
```

## 📝 SMILES Examples

```
Aspirin (Acetic acid derivative):   CC(=O)OC(CC(=O)O)C[N+](C)(C)C
Ethanol:                            CCO
Benzene:                            c1ccccc1
Caffeine:                           CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Toluene:                            Cc1ccccc1
Acetone:                            CC(=O)C
Phenol:                             Oc1ccccc1
Naphthalene:                        c1cc2ccccc2cc1
```

## 📚 References

- Morgan Fingerprints: Rogers & Hahn (2010)
- logP Prediction: Wildman & Crippen (1999)
- PyTorch: Paszke et al. (2019)
- RDKit: Landrum et al.

## 🤝 Contributing

To contribute improvements:

1. Fork the repository
2. Create a branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📄 License

MIT License - see LICENSE file

## 👨‍💻 Author

[Your Name]
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile]
- Email: your.email@example.com

## 🙏 Acknowledgments

- **PubChem** - for the dataset
- **RDKit** - for cheminformatics tools
- **PyTorch** - for deep learning framework
- **Streamlit** - for web framework

## 📞 Support

If you have questions or found a bug:
- Open an Issue on GitHub
- Send an email
- Contact via LinkedIn

## 🔮 Future Improvements

- [ ] Batch prediction support
- [ ] Export results to CSV
- [ ] Molecular structure visualization
- [ ] Model uncertainty estimation
- [ ] Hyperparameter optimization UI
- [ ] Model versioning
- [ ] FastAPI endpoint
- [ ] Docker support
- [ ] Multi-language support
- [ ] Comparison with other models

---

**Last Updated:** December 2025