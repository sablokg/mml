# Real benchmark: Zhu et al. 2020 gut metagenomes (schizophrenia vs healthy controls)

Data from [Zhu et al. 2020, *Nature Communications* 11:1612](https://www.nature.com/articles/s41467-020-15457-9); MetaPhlAn read counts and sample metadata as distributed with the [ML-for-microbiomics tutorial](https://github.com/quantori/tutorial-on-ML-for-microbiomics). 171 samples (90 SCZ, 81 HC).

| File | Purpose |
|---|---|
| `data/genus.tsv`, `data/species.tsv` | raw-count tables (sample id, taxa..., `__label__`); 543 genera, 1082 species |
| `prep.py` | rebuilds the tables: `python3 prep.py <tutorial dir>` |
| `zhu_bench.rs` | Rust runner (cargo example `zhu_bench`, needs `--features xgboost`); repeated stratified 5-fold CV, 7 fixed pipelines + AUTO |
| `sk_ref.py` | scikit-learn CLR + LR / RF on the same folds (`results/folds_*.csv`) |
| `analyze.py` | mean (SD), paired Wilcoxon vs CLR + LR, AUTO choices |
| `run_all.sh` | runs everything (`./run_all.sh [reps=3] [auto=1]`) |
| `results/` | per-fold scores (`res_*.csv`, `sk_*.csv`), folds, `summary_*.csv`, `auto_choices_*.csv` |

Run from this directory: `./run_all.sh`. Fixed seeds (1000 + repeat) give identical folds and scores; timings vary.

## Results (balanced accuracy, mean (SD) over 3 x 5 folds)

| Pipeline (balanced accuracy, mean (SD) over 15 folds) | Genus (543 taxa) | Species (1082 taxa) | s/fit (genus / species) |
|---|---|---|---|
| **CLR + logistic regression** (microbiome-ml) | **0.712 (0.084)** | **0.800 (0.068)** | 0.02 / 0.04 |
| CLR + logistic regression (scikit-learn) | 0.714 (0.083) | 0.802 (0.071) | 0.006 / 0.006 |
| AUTO (`AutoClassifier::default()`) | 0.702 (0.092) | 0.790 (0.071) | 25.5 / 45.6 |
| ILR (data-driven) + logistic regression | 0.698 (0.078) | 0.781 (0.069) | 0.02 / 0.06 |
| Presence/absence + random forest | 0.708 (0.097) | 0.744 (0.071) | 0.85 / 1.5 |
| CLR + XGBoost | 0.670 (0.083) | 0.731 (0.069) | 0.4 / 0.8 |
| log-TSS + random forest | 0.676 (0.073) | 0.716 (0.066) | 0.96 / 1.8 |
| CLR + random forest (microbiome-ml) | 0.672 (0.078) | 0.714 (0.065) | 1.1 / 2.2 |
| TSS + random forest | 0.678 (0.069) | 0.708 (0.063) | 0.95 / 1.8 |
| CLR + random forest (scikit-learn) | 0.679 (0.076) | 0.692 (0.055) | 0.14 / 0.16 |

Rows named `AUTO:choice=...` in `results/res_*.csv` are bookkeeping for which pair AUTO selected (score column is 0 by construction); `analyze.py` excludes them. See the main README for interpretation and caveats.