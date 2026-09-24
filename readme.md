# 🧬 logP Predictor

Deep learning application predicting molecular lipophilicity (logP) from SMILES structure, using a PyTorch neural network trained on hybrid molecular fingerprint + descriptor features.

**Live demo:** https://molecular-lipophilicity-ai.streamlit.app/

## About the project

**logP (partition coefficient)** is the logarithm of a molecule's distribution between octanol and water — a key descriptor in chemistry, pharmacy and biotechnology, driving:

- Water vs. fat solubility
- Biological membrane permeability
- Drug bioavailability
- Molecular hydrophobicity

## Model

| Parameter | Value |
|---|---|
| Model | 4-layer MLP (PyTorch) |
| Architecture | 2054 → 512 → 256 → 128 → 1, BatchNorm + ReLU + Dropout(0.3) per layer |
| Parameters | 1,218,305 |
| Features | 2048-bit Morgan fingerprints + 6 physicochemical descriptors |
| Training data | 11,612 molecules from PubChem |
| Training | 72/100 epochs (early stopping), 8m 51s |

**Results:**

| Split | R² | MAE |
|---|---|---|
| Train | 0.9948 | 0.1411 |
| Validation | 0.9093 | 0.6045 |
| Test | 0.9007 | 0.6166 |

The train/validation/test gap is the expected signature of a ~1.2M-parameter model on ~11.6k molecules — addressed with dropout, batch norm and early stopping (training stopped at epoch 72/100, before the gap widened further) rather than chased away with more capacity.

## Tech stack

- **Modeling:** PyTorch, scikit-learn (preprocessing), RDKit (Morgan fingerprints, descriptors)
- **App:** Streamlit
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Community Cloud

## Project structure

```
molecular-lipophilicity-ai/
├── app.py                          # Streamlit application (the deployed entrypoint)
├── EDA.ipynb                       # Exploratory data analysis
├── pytorch_logP_pred.ipynb         # Feature engineering, model training, evaluation
├── fixing.ipynb                    # PyTorch → ONNX model export
├── checkpoints/
│   └── best_model.pt               # Trained PyTorch model (best epoch)
├── data/
│   ├── raw/RAW.csv                 # Raw PubChem data
│   └── procced/                    # Preprocessed data + scaler params
├── visualisation/                  # EDA and training plots
├── requirements.txt
└── LICENSE
```

## How it works

1. Enter a SMILES string (manually, or pick from the built-in examples)
2. RDKit validates the structure and computes a 2048-bit Morgan fingerprint plus 6 molecular descriptors (molecular weight, TPSA, H-bond donors/acceptors, rotatable bonds, etc.)
3. The trained MLP predicts logP from the combined 2054-dim feature vector
4. The app shows the predicted value with a color-coded lipophilicity interpretation

| logP | Interpretation |
|---|---|
| < -2 | Very hydrophilic 🔵 |
| -2 to 0 | Hydrophilic 🟢 |
| 0 to 2 | Moderate (optimal for oral drugs) 🟡 |
| 2 to 5 | Lipophilic 🟠 |
| > 5 | Highly lipophilic 🔴 |

## Running locally

```bash
git clone https://github.com/slastrzelec/molecular-lipophilicity-ai.git
cd molecular-lipophilicity-ai
pip install -r requirements.txt
streamlit run app.py
```

## Training the model

```bash
jupyter notebook pytorch_logP_pred.ipynb
```

Covers EDA, Morgan fingerprint + descriptor feature engineering, model training with early stopping, and evaluation.

## Example SMILES

```
Aspirin:      CC(=O)Oc1ccccc1C(=O)O
Ethanol:      CCO
Benzene:      c1ccccc1
Caffeine:     CN1C=NC2=C1C(=O)N(C(=O)N2C)C
Toluene:      Cc1ccccc1
Acetone:      CC(=O)C
Phenol:       Oc1ccccc1
Naphthalene:  c1cc2ccccc2cc1
```

## Dataset

Source: [PubChem](https://pubchem.ncbi.nlm.nih.gov/) — 11,612 molecules with experimental/computed logP values. Preprocessing: outlier removal, StandardScaler normalization.

## References

- Rogers, D. & Hahn, M. (2010). Extended-connectivity fingerprints.
- Wildman, S. A. & Crippen, G. M. (1999). Prediction of physicochemical parameters by atomic contributions.
- Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library.
- [RDKit documentation](https://www.rdkit.org/)

## License

MIT License.
