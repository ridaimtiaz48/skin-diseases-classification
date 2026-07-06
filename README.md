# Skin Disease Classification — Hybrid Vision Transformer + Quantum Neural Network

A 6-class skin disease classification pipeline combining a pretrained **Vision Transformer (ViT-B/16)** feature extractor with a **6-qubit variational quantum circuit**, achieving **97.80% test accuracy** across six visually-similar pox-type skin conditions: Chickenpox, Cowpox, HFMD, Healthy, Measles, and Monkeypox.

## Architecture

The pipeline has 3 stages, connected end-to-end:

1. **Stage 1 — Vision Transformer (Feature Extractor)**
   - ViT-B/16, pretrained on ImageNet-1K
   - Last 2 encoder blocks fine-tuned; rest frozen (prevents overfitting on a small medical dataset)
   - Outputs a 768-dim CLS token → reduced to 6-dim via a linear layer + BatchNorm

2. **Stage 2 — Quantum Circuit**
   - 6 qubits (matching the 6 output classes), 2 StronglyEntanglingLayers
   - Classical features encoded as qubit rotation angles (`sigmoid(x) * π → RY gates`)
   - Simulated using PennyLane's `lightning.qubit` backend with batched processing for speed
   - Outputs 6 Pauli-Z expectation values

3. **Stage 3 — Classifier**
   - Linear layer mapping quantum circuit outputs → 6 disease classes

```
Image → ViT-B/16 (frozen + fine-tuned) → Linear(768→6) → BatchNorm → Quantum Circuit (6 qubits) → Linear(6→6) → Prediction
```

## Key Techniques Used

- Transfer learning with differential learning rates (backbone vs. new layers)
- BatchNorm before quantum angle encoding for stable input range
- Weighted CrossEntropyLoss with label smoothing to handle class imbalance
- WeightedRandomSampler for balanced batch construction
- AdamW optimizer + CosineAnnealingLR scheduler
- Early stopping with best-checkpoint restoration
- Data augmentation: rotation, flips, color jitter, cutout/random erasing
- Explainable AI (XAI): prediction confidence grid, ViT attention overlay (CLS-patch similarity), quantum output distribution heatmap per class

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 97.80% |
| Best Validation Accuracy | 98.14% (epoch 44) |
| Final Train Accuracy | 98.79% |
| Train/Val Gap | 0.82% |
| Macro Precision | 0.9707 |
| Macro Recall | 0.9845 |
| Macro F1-Score | 0.9774 |

**Per-class performance:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Chickenpox | 0.96 | 0.98 | 0.97 | 113 |
| Cowpox | 0.95 | 1.00 | 0.98 | 99 |
| HFMD | 1.00 | 0.98 | 0.99 | 242 |
| Healthy | 0.98 | 0.98 | 0.98 | 171 |
| Measles | 0.95 | 1.00 | 0.98 | 83 |
| Monkeypox | 0.98 | 0.97 | 0.98 | 426 |

*Trained for 50 epochs (early stopping patience=8, best model at epoch 44) at 224×224 resolution. Batch size 32, AdamW optimizer with differential learning rates (backbone LR=3e-5, new layers LR=3e-4), CosineAnnealingLR schedule.*

## Dataset

- 6-class image dataset distinguishing Chickenpox, Cowpox, HFMD (Hand, Foot & Mouth Disease), Healthy skin, Measles, and Monkeypox — visually similar pox-type skin conditions that are historically difficult to distinguish clinically
- 1,134 test images total, with class sizes ranging from 83 to 426 samples
- Proper train/validation/test split with no data leakage
- Class imbalance handled via inverse-frequency class weighting and WeightedRandomSampler

## Explainability (XAI)

The notebook includes three interpretability visualizations:
- **Prediction grid** — sample predictions with confidence scores
- **Attention overlay** — CLS-token to patch-token similarity from the last ViT block, visualized as a heatmap over the original image
- **Quantum output heatmap** — mean Pauli-Z expectation values per class across all 6 qubits, showing how the quantum layer separates classes

## Tech Stack

`Python` · `PyTorch` · `torchvision` · `PennyLane` (+ `pennylane-lightning`) · `Vision Transformer (ViT-B/16)` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Google Colab (GPU)`

## Repository Structure

```
├── Skin_Disease_Classification.ipynb   # Main notebook (end-to-end pipeline)
├── README.md
├── requirements.txt
├── .gitignore
└── outputs/
    ├── confusion_matrix.png
    ├── prediction_grid.png
    ├── attention_maps.png
    └── quantum_output_heatmap.png
```

## How to Run

1. Open the notebook in Google Colab (GPU runtime — T4 or better recommended)
2. Mount your Google Drive and update `DATASET_ROOT` in the Global Configuration section to point to your dataset
3. Run all cells sequentially — dependencies (PennyLane, pennylane-lightning, seaborn) are installed automatically
4. Training progress, evaluation metrics, and XAI visualizations are generated and saved automatically

## Future Improvements

- Benchmark against a classical-only baseline (ViT without the quantum layer) to isolate the quantum circuit's actual contribution
- Test on real quantum hardware (e.g., via IBM Quantum) instead of simulation
- Expand dataset size / class coverage for better generalization

## License

This project was developed for academic purposes.
