# CausalPrep — ColoredMNIST

Preprocessing-based causal feature filtering for distribution-shift robustness, evaluated on the ColoredMNIST benchmark.

## What this does

ColoredMNIST is built so that digit **color** is a spurious shortcut: in training, color is strongly correlated with the label, but at test time that correlation is flipped. A model that learns color instead of digit **shape** will do well in training and badly on test.

This script runs a full causal preprocessing pipeline that:
1. Discovers which engineered features are confounded by color using causal graph discovery (FCI),
2. Computes a provably valid adjustment set for each confounded feature (or correctly reports when none exists),
3. Removes the color-driven part of each feature via residualization,
4. Estimates each feature's actual causal effect on the label,
5. Filters out features that don't pass causal + statistical criteria,
6. Trains a downstream classifier on the filtered features and compares it against a classifier trained on raw features, evaluating both on the anti-correlated test set.

## Pipeline steps

| Step | What happens |
|---|---|
| 1. Data loading | Loads `train1.pt`, `train2.pt`, `test.pt`. train1+train2 are combined for training; test is held out for OOD evaluation. |
| 2. Feature extraction | Extracts 7 features per image: 6 shape features (causal) + 1 color feature (spurious). |
| 3. FCI causal discovery | Runs FCI on features + color to learn a PAG (Partial Ancestral Graph) and flag which features are confounded by color. |
| 4. Generalized Adjustment Criterion (GAC) | For each confounded feature, searches the PAG for a provably valid adjustment set Z. If none exists, the feature is marked `not_identifiable` and left unadjusted rather than guessed at. |
| 5. Residualization | Regresses each adjusted feature on its discovered Z and keeps only the residual (the part not explained by Z). |
| 6. ACE estimation | Estimates each feature's Average Causal Effect on the label via doubly-robust double machine learning (DML). |
| 7. Feature filtering | Keeps a feature only if it passes all of: `\|ACE\| ≥ 0.05`, `p < 0.05`, `R² with color < 0.10`, and "not flagged confounded by FCI". |
| 8. Filtered feature matrix | Builds the final train/test matrices from the kept, residualized features. |
| 9. Downstream models | Trains two logistic regression models — **Raw** (all 7 features) and **CausalPrep** (filtered + residualized features) — and evaluates both on the anti-correlated test set. |
| 10. IRM-style diagnostics | Measures per-group accuracy, worst-group accuracy, and an IRM-style invariance penalty per environment, comparing Raw vs. CausalPrep. |

## Features

| Feature | Type | Description |
|---|---|---|
| `hog_energy_shape` | causal | Sum of squared HOG descriptor values (gradient energy) |
| `hog_entropy_shape` | causal | Entropy of the HOG descriptor (texture complexity) |
| `edge_density_shape` | causal | Fraction of pixels with high gradient magnitude |
| `stroke_width_shape` | causal | Mean horizontal run-length of bright (digit) pixels |
| `digit_area_shape` | causal | Fraction of pixels classified as digit |
| `aspect_ratio_shape` | causal | Bounding-box height/width of the digit region |
| `mean_red_color` | spurious | Mean red-channel intensity of digit pixels |

## Why GAC instead of a fixed `Z = {color}`

A PAG can contain **circle marks** (`o`) — edges whose direction FCI couldn't fully resolve from the data. A simpler approach might just adjust on `color` whenever the PAG shows *any* link to it, but that can be wrong: color might be a descendant of the feature (adjusting on it would introduce collider bias), or a different variable might be needed for the adjustment to be valid at all.

The Generalized Adjustment Criterion (GAC), based on Maathuis & Colombo (2015), instead:
- Computes which nodes are forbidden from any adjustment set (possible descendants of the feature on a path to color),
- Searches over the other measured variables for a set that blocks every relevant path under **every** way the PAG's circle marks could resolve,
- Returns the smallest valid set if one exists, or reports the feature as **not identifiable** from this PAG, leaving it unadjusted rather than residualizing on an unjustified guess.

## Requirements

```
numpy
matplotlib
torch
tqdm
scikit-image
scikit-learn
scipy
causal-learn
```

Install with:
```bash
pip install numpy matplotlib torch tqdm scikit-image scikit-learn scipy causal-learn
```

## Usage

```bash
python3 causalprep_coloredmnist1.py --data_dir /path/to/data --output_dir ./results
```

`--data_dir` must contain `train1.pt`, `train2.pt`, and `test.pt` (the standard ColoredMNIST `.pt` splits).

unzip the data zip folder

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | *(required)* | Directory containing `train1.pt`, `train2.pt`, `test.pt` |
| `--output_dir` | `./results` | Where to write the results plot |
| `--max_samples` | `None` | Subsample each split for quick testing |
| `--fci_samples` | `2000` | Subsample size used for the FCI run (FCI cost grows with sample size) |
| `--alpha` | `0.05` | FCI significance level (Fisher-Z test) |

## Output

- Console output: a full step-by-step log (feature extraction stats, FCI edges, GAC adjustment sets per feature, ACE estimates, filtering decisions, model accuracy/AUC, IRM-style diagnostics, and a final diagnostic report).
- `causalprep_coloredmnist_results.png` in `--output_dir`: a 6-panel figure with Raw vs. CausalPrep test accuracy, AUC-ROC, per-feature R² with color, ACE estimates, per-group accuracy, and IRM penalty per environment.

## Notes

- `ColoredMNIST`'s spurious signal is a single scalar (color), so the GAC candidate pool for `mean_red_color` is just the 6 shape features. If FCI doesn't find a usable proxy among them, this feature will typically come back `not_identifiable` — that's the expected, honest outcome of the criterion, not a bug.
- Residualization for the test split fits its regression on **train only** and applies it to test, to avoid leaking test-set color distribution into the adjustment.
