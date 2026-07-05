# KlaCC
BIBM program
# Protein Sequence Classification with Transformer + Contrastive Learning

This repository implements a deep learning pipeline for binary classification of protein sequences (e.g., predicting functional sites). The model combines a Transformer encoder with a projection head and a classification head, trained using a three‑stage strategy: BCE warm‑up, supervised contrastive learning (SupCon), and end‑to‑end fine‑tuning. The code performs 5‑fold cross‑validation for hyperparameter selection, then trains a final model on the full training set and evaluates it on an independent test set. It also generates t‑SNE visualizations and sequence logos for interpretability.

---

## Features

- **Transformer backbone** with learnable positional embeddings and amino‑acid embedding.
- **Projection head** for contrastive representation learning.
- **Three‑stage training**:
  1. BCE warm‑up (supervised)
  2. SupCon pre‑training (separation‑based early stopping)
  3. End‑to‑end fine‑tuning (optional freeze of encoder)
- **5‑fold cross‑validation** with AUC‑based early stopping.
- **Independent test set evaluation** with comprehensive metrics (AUC, AUPRC, F1, MCC, specificity, etc.).
- **Visualizations**:
  - ROC / PR curves for each fold and test set.
  - Confusion matrix.
  - t‑SNE plots colored by true labels and KMeans clusters.
  - Sequence logos for each cluster and class.

---

## Requirements

- Python 3.9
- PyTorch 2.0+
- NumPy, pandas, scikit‑learn, matplotlib, seaborn
- logomaker (for sequence logos)
- openpyxl (for Excel file reading)

All dependencies are listed in `environment.yml`. Use Conda to set up the environment.

---

## Data Preparation

The script expects three files:

1. **`train.xlsx`** – training data with columns:
   - `protein_sequence` : full protein sequence (string)
   - `label` : binary label (0 or 1)
   - `position` : 1‑based index of the central residue (the site of interest)

2. **`test.xlsx`** – same format as training, used for final evaluation.

3. **`aa_embeddings.csv`** – (optional) pre‑trained amino‑acid embeddings; if not provided, the code will initialize random embeddings of size 64. The current implementation does not actually use the CSV content, but the path is kept for future extension.

> **Note**: The script extracts a window of length `SEQ_LENGTH` (default 51) around the given position. Positions outside the sequence are padded with `'*'` (left) and `'#'` (right).

---

## Configuration

All hyperparameters are defined in the `HyperParams` class inside `train.py`. You can modify them directly in the script or override via command‑line arguments.

Key adjustable parameters:

| Parameter | Description |
|-----------|-------------|
| `SEQ_LENGTH` | Window size around the central position (default 51). |
| `TRANSFORMER_D_MODEL`, `TRANSFORMER_LAYERS` | Transformer dimensions and depth. |
| `UNIFIED_DIM` | Dimension of the Transformer output (default 512). |
| `PROJECTION_OUT_DIM` | Dimension of the projection head output (default 128). |
| `USE_STAGE1 / USE_STAGE2 / USE_STAGE3` | Enable/disable each training stage (ablation study). |
| `N_FOLDS` | Number of cross‑validation folds (default 5). |
| `STAGE*_EPOCHS`, `STAGE*_LR`, `STAGE*_BATCH_SIZE` | Per‑stage training settings. |
| `OUTPUT_DIR` | Directory where all results and plots are saved. |

---

## Usage

### 1. Prepare your data

Place `train.xlsx`, `test.xlsx` (and optionally `aa_embeddings.csv`) in the project root, or adjust the paths in `HyperParams`.

### 2. Run the pipeline

```bash
python train.py
Command‑line arguments
Argument	Description
--train_path	Override training data path.
--test_path	Override test data path.
--output_dir	Override output directory.
--n_folds	Number of CV folds.
--no_stage1	Skip BCE warm‑up.
--no_stage2	Skip SupCon pre‑training.
--no_stage3	Skip end‑to‑end fine‑tuning.
Example:

bash
python train.py --train_path ./data/my_train.xlsx --test_path ./data/my_test.xlsx --output_dir ./results/exp1 --n_folds 5
Output
All results are saved in OUTPUT_DIR (default ./experiments/cv_test_final). The directory contains:

config.json – full hyperparameter configuration.

cv_results.json – AUC scores for each CV fold.

test_results.json – final test metrics and summary.

final_model.pth / final_encoder.pth – saved model weights.

plots/ – subfolder with all visualizations:

ROC/PR curves per fold and for test set.

CV AUC bar chart.

Confusion matrix.

t‑SNE plots (by label and by cluster).

Sequence logos for each cluster and each class.

Key Functions
train_stage1() : BCE warm‑up with early stopping.

train_stage2() : SupCon training with separation‑based early stopping.

train_stage3() : Fine‑tuning the classifier (encoder may be frozen).

evaluate() : Computes all metrics and returns features for visualization.

compute_separation() : Measures the difference between mean positive and negative cosine similarities.

perform_tsne_and_clustering() : Runs t‑SNE and KMeans clustering on the projected features.

Customization
To change the loss function or add new metrics, modify compute_all_metrics().

To use a different backbone, replace the TransformerEncoder class.

The classifier input is a concatenation of backbone features and projection features – adjust UNIFIED_DIM and PROJECTION_OUT_DIM accordingly.
