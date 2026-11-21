# microbiome-ml

Automated machine learning for microbiome abundance tables in Rust. Requires Rust 1.85+ (edition-2024 dependency `smelt-ml`).

## Features

- **Transforms** (all fitted on training data only): prevalence/abundance filtering, TSS, log(TSS+pc), CLR, arcsin-sqrt, presence/absence, optional z-scoring, plus
  - **ILR balances** (`Transform::Ilr`): phylogenetic when a tree is supplied (PhILR-style topology, no coordinate weighting; polytomies resolved arbitrarily), otherwise a data-driven dendrogram (average linkage on the training variation matrix; up to 2000 taxa). Balances are an isometry of CLR space (unit-tested).
  - **Clade-aggregated CLR** (`Transform::CladeClr`): sum taxa into maximal clades below a height cut (fraction of tree height), then CLR. Needs branch lengths.
  - Newick parsing (`PhyloTree::from_newick`), tree-to-table matching by taxon name; taxa absent from the tree are dropped for phylogenetic transforms and counted in `Preprocessor::unplaced`.
- **Models:** classification: random forest, logistic regression, decision tree, kNN (smartcore). Regression: random forest, elastic net, decision tree (smartcore), ridge (own dual-form implementation, leave-one-out penalty, valid for p > n). **XGBoost** (classification and regression) via the pure-Rust [`smelt-ml`](https://crates.io/crates/smelt-ml) crate, behind the `xgboost` feature; no C++ toolchain.
- **Automated search** (`AutoClassifier` / `AutoRegressor`): cross-validates every transform x model pair; optional **hyperparameter grid** (`.with_grid(GridLevel::Small | Full)`) over the `refine_top` best pairs; `.with_phylogeny(tree, &feature_names)` adds the phylogenetic transforms; failed candidates are listed in the report (`FAILED (excluded)`).
- **Feature importance mapped to taxa:** `taxon_importance(x, y, n_repeats, seed)` = permutation importance through the whole fitted pipeline, reported per raw taxon name (works even when the model sees balances or clades). `native_importance()` returns XGBoost gain importances labelled by output feature (taxon, balance or clade).

```rust
use microbiome_ml::prelude::*;
let data = Dataset::from_delimited("table.tsv", '\t', false)?;
let tree = PhyloTree::from_newick_file("tree.nwk")?;
let fit = AutoClassifier::default()
    .with_phylogeny(tree, &data.feature_names)
    .with_grid(GridLevel::Small)
    .fit(&data, &labels)?;
println!("{}", fit.report());
let imp = fit.taxon_importance(&held_out_x, &held_out_labels, 10, 1)?; // use held-out data
println!("{}", format_importance(&imp, 15));
```

`cargo test` (add `--features xgboost` for the XGBoost tests): 11 unit + 5 integration tests + doctest.

## Repository layout

`src/` library, `examples/` and `benchmarks/` simulated-data benchmarks, `real_benchmark/` real-data benchmark (Zhu et al. 2020: tables, runner `zhu_bench.rs`, scikit-learn reference, results), `tests/`.

## What the benchmarks show

Simulated cohorts (`benchmarks/`) and one real cohort (`real_benchmark/`, below).

### Simulated data

- **Auto search vs best single pipeline (v0.1 benchmark, 40 datasets/task):** ties it (balanced accuracy 0.830 vs 0.836; R2 0.272 vs 0.282; p > 0.3).
- **v0.2 feature benchmark** (n=120, p=100, 8 replicates, held-out test; `feature_bench_results.csv`):
  - **XGBoost (smelt)** is competitive: CLR+XGB 0.950 / 0.919 vs CLR+RF 0.922 / 0.922 (clade / scattered signal; differences not significant) and about 2x faster per fit than the smartcore forest here. It was the most frequently selected model by the auto search. Not compared against the C++ XGBoost library.
  - **Hyperparameter grid:** no measurable test-set gain (0.941 vs 0.941 and 0.938 vs 0.931), at about 1.9-2.8x the search time. At n=80 training samples the inner CV rarely preferred a non-default setting.
  - **ILR and clade transforms did not help**, even when the signal is a single clade: ILR (phylogenetic) + logistic regression scored 0.844 vs CLR 0.900 (p = 0.02); data-driven ILR + LR 0.809 vs 0.900 (p = 0.008); with random forests ILR was neutral to worse. The simulated tree is not tied to abundance structure, so this does not test whether phylogenetic transforms help on real, phylogenetically structured cohorts; it only shows they are not a free win.
  - **Importance:** top-6 precision for recovering the 6 true signal taxa (chance about 0.06): XGBoost native 0.90 / 0.94; permutation on a 40-sample test set 0.65-0.79 (noisy at that size).

### Real-benchmark directory (`real_benchmark/`)

Zhu et al. data: MetaPhlAn read counts and sample metadata as distributed; 171 samples (90 SCZ, 81 HC). The 20 further datasets come from the public metageomic_datasets collection.

| File | Purpose |
|---|---|
| `data/genus.tsv`, `data/species.tsv` | Zhu raw-count tables (sample id, taxa..., `__label__`); 543 genera, 1082 species |
| `data/<dataset>.tsv`, `data/datasets_summary.csv` | the 20 further datasets (relative abundances, `__label__` = control / disease) and their sizes |
| `prep.py` | rebuilds the Zhu tables: `python3 prep.py <tutorial dir>` |
| `prep_metagenomic.py` | rebuilds the 20 tables: `python3 prep_metagenomic.py <metageomic_datasets dir>` |
| `zhu_bench.rs` | Rust runner (cargo example `zhu_bench`, needs `--features xgboost`); repeated stratified 5-fold CV, 7 fixed pipelines + AUTO |
| `sk_ref.py` | scikit-learn CLR + LR / RF on the same folds (`results/folds_*.csv`); `python3 sk_ref.py <dataset>` |
| `analyze.py` | mean (SD), paired Wilcoxon vs CLR + LR, AUTO choices; accepts dataset names; across datasets also writes `results/benchmark_matrix_mean.csv`, `benchmark_ranks.csv`, `benchmark_all_long.csv`, `analysis_log.txt` |
| `run_all.sh` | Zhu benchmark (`./run_all.sh [reps=3] [auto=1]`) |
| `run_metagenomic.sh` | the 20 further datasets (`./run_metagenomic.sh [reps=3] [auto=0] [dataset ...]`; about 2 h single-threaded; finished datasets are skipped) |
| `results/` | per-fold scores (`res_*.csv`, `sk_*.csv`), folds, `summary_*.csv`, `auto_choices_*.csv` |

Run from `real_benchmark/`. Fixed seeds (1000 + repeat) give identical folds and scores; timings vary. Rows named `AUTO:choice=...` in `results/res_*.csv` are bookkeeping for which pair AUTO selected (score column is 0 by construction); `analyze.py` excludes them.

### Real data: gut metagenomes in schizophrenia

Benchmark on the cohort of [Zhu et al. 2020, *Nature Communications* 11:1612](https://www.nature.com/articles/s41467-020-15457-9) (MetaPhlAn counts, 90 schizophrenia cases vs 81 healthy controls, n = 171), via the ML-for-microbiomics tutorial data. Repeated stratified 5-fold CV (3 repeats = 15 outer folds); all preprocessing is fitted inside each training fold, and AUTO's selection is nested in the training fold. Chance is 0.5. Code, data and results are in [`real_benchmark/`](real_benchmark/) (`./real_benchmark/run_all.sh`; files listed above).

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

- **CLR + logistic regression was best at both levels**, and the Rust implementation matches scikit-learn on the identical folds to within 0.002 (paired Wilcoxon p = 0.32, both levels). Species-level features beat genus-level ones by about 9 points for the leading models.
- **Forests and XGBoost trailed CLR + LR.** At species level the gap was 0.06 to 0.09 (p = 0.001 to 0.005, including presence/absence + RF). At genus level it was 0.03 to 0.04 for forests and XGBoost (p = 0.03 to 0.07), while presence/absence + RF was level. Rust and scikit-learn forests differ by -0.007 (genus) and +0.022 (species), neither significant (p = 0.59, 0.09); scikit-learn uses its default `sqrt(p)` features per split, microbiome-ml uses all.
- **AUTO tied CLR + LR (-0.010 at both levels, p = 0.36 to 0.64) but cost ~1,000x the time and was unstable in its choice**: 10 different transform/model pairs across 15 genus folds (most common: presence/absence + XGBoost, 4); at species level log-TSS + logistic regression 9/15. This agrees with the simulated result that the search avoids poor pipelines but does not beat a well-chosen default.
- **Data-driven ILR + LR did not help** (-0.014 genus, -0.019 species; p = 0.31, 0.09), consistent with the simulations.

Caveats: one cohort; the 15 folds come from 3 repeats on the same 171 samples, so they are not independent and p-values are unadjusted and indicative only; replicate SD is 0.06 to 0.10, so differences under about 0.03 are within noise.

### Real data: 20 public metagenomic datasets

All datasets of the metageomic_datasets collection of public low-sample-size human gut metagenomic sets (Disease-sets-1, Disease-sets-2, CRC-Nation, CRC-Stage), each as a binary disease-vs-control task. Same protocol as the Zhu benchmark above (3 x 5-fold stratified CV, seeds 1000 + repeat, preprocessing fitted inside training folds, scikit-learn on the identical folds) with the seven fixed pipelines; **AUTO was not run** on these datasets (it costs tens of seconds per fit even on the smallest sets and far more on the CRC-Stage tables with thousands of taxa). Chance is 0.5.

Notes on the data:
- Inputs are **relative abundances** (percent or fraction), not raw counts; the preprocessor rescales to total-sum scaling anyway, so the pipelines run unchanged.
- Disease-sets-1 CRC: the 39 adenoma samples are excluded (cancer 48 vs healthy 47). IBD: ulcerative colitis (21) and Crohn's disease (4) are pooled into one class of 25 cases, as in the source.
- CRC-Stage: each stage (MP, S0, SI/II, SIII/IV) is compared with the same 127 healthy controls. About 2,800 taxa pass the prevalence filter, which exceeds the 2,000-taxa limit of data-driven ILR, so **ILR (data-driven) is not applicable (n/a) on the four CRC-Stage tasks** (`*` rows cover 16 datasets only and are not rank-comparable).
- Disease-sets-1 and Disease-sets-2 draw on the same cohorts for Cirrhosis, IBD and Obesity (largely identical samples), so these tasks are not independent. The CRC-Stage tasks share their controls.
- The Zhu results above are not pooled into the cross-dataset summary.

#### Balanced accuracy per dataset (mean over 15 folds; best per row in bold; SDs in `results/summary_<dataset>.csv`)

| Dataset | n (disease/control) | taxa | CLR+LR | CLR+LR (sk) | ILR+LR | CLR+RF | CLR+RF (sk) | CLR+XGB | LogTSS+RF | TSS+RF | PA+RF |
|---|---|---|---|---|---|---|---|---|---|---|---|
| DS1_Cirrhosis | 232 (118/114) | 542 | **0.878** | 0.877 | 0.870 | 0.851 | 0.873 | 0.865 | 0.845 | 0.844 | 0.866 |
| DS1_IBD | 110 (25/85) | 443 | **0.689** | **0.689** | 0.673 | 0.656 | 0.561 | 0.660 | 0.671 | 0.671 | 0.672 |
| DS1_Obesity | 253 (164/89) | 465 | 0.563 | 0.565 | 0.572 | 0.586 | **0.608** | 0.578 | 0.524 | 0.534 | 0.535 |
| DS1_T2D | 440 (223/217) | 606 | 0.623 | 0.624 | 0.632 | 0.645 | 0.643 | 0.636 | 0.643 | 0.643 | **0.658** |
| DS1_CRC | 95 (48/47) | 492 | 0.730 | 0.726 | 0.727 | **0.781** | **0.781** | 0.727 | 0.765 | 0.764 | 0.730 |
| DS2_Cirrhosis | 232 (118/114) | 542 | **0.878** | 0.877 | 0.870 | 0.847 | 0.875 | 0.865 | 0.845 | 0.842 | 0.867 |
| DS2_IBD | 110 (25/85) | 443 | **0.689** | **0.689** | 0.673 | 0.663 | 0.586 | 0.669 | 0.675 | 0.669 | 0.676 |
| DS2_Obesity | 253 (164/89) | 465 | 0.563 | 0.565 | 0.572 | **0.598** | 0.595 | 0.581 | 0.524 | 0.528 | 0.524 |
| DS2_T2D | 344 (170/174) | 572 | 0.651 | 0.652 | 0.636 | 0.693 | **0.698** | 0.681 | 0.680 | 0.684 | 0.661 |
| DS2_WT2D | 96 (53/43) | 381 | 0.719 | 0.719 | 0.675 | **0.739** | 0.671 | 0.724 | 0.642 | 0.646 | 0.731 |
| DS2_CRC | 121 (48/73) | 503 | 0.716 | 0.715 | 0.679 | **0.784** | 0.749 | 0.738 | 0.761 | 0.763 | 0.706 |
| CRCNation_AUS | 109 (46/63) | 828 | 0.856 | 0.856 | 0.835 | 0.789 | 0.787 | 0.816 | 0.797 | 0.797 | **0.860** |
| CRCNation_CHN | 128 (74/54) | 762 | 0.763 | 0.766 | 0.744 | 0.779 | 0.759 | **0.784** | 0.780 | 0.783 | 0.746 |
| CRCNation_DEU | 120 (60/60) | 840 | 0.794 | 0.792 | 0.786 | 0.733 | 0.711 | 0.750 | 0.797 | 0.792 | **0.806** |
| CRCNation_FRA | 114 (53/61) | 842 | 0.745 | 0.745 | 0.755 | **0.773** | 0.756 | 0.702 | 0.755 | 0.758 | 0.745 |
| CRCNation_USA | 104 (52/52) | 828 | **0.635** | **0.635** | 0.579 | 0.568 | 0.561 | 0.571 | 0.562 | 0.559 | 0.558 |
| CRCStage_MP | 167 (40/127) | 6370 | **0.557** | **0.557** | n/a | 0.496 | 0.506 | 0.547 | 0.495 | 0.495 | 0.502 |
| CRCStage_S0 | 154 (27/127) | 6287 | 0.565 | **0.567** | n/a | 0.495 | 0.500 | 0.507 | 0.506 | 0.500 | 0.496 |
| CRCStage_SI-II | 196 (69/127) | 6604 | 0.580 | 0.583 | n/a | **0.589** | 0.554 | 0.576 | 0.572 | 0.574 | 0.513 |
| CRCStage_SIII-IV | 181 (54/127) | 6475 | 0.639 | 0.639 | n/a | 0.675 | 0.561 | **0.686** | 0.673 | 0.673 | 0.544 |

#### Summary over the 20 datasets

| Pipeline | datasets | mean bal. acc. | mean rank (of 8) | wins | mean diff vs CLR+LR | better / worse than CLR+LR | median s/fit |
|---|---|---|---|---|---|---|---|
| CLR + random forest (microbiome-ml) | 20 | 0.687 | 3.83 | 5 | -0.005 | 11 / 9 | 1.58 |
| CLR + logistic regression (scikit-learn) | 20 | 0.692 | 3.88 | 1 | +0.000 | 7 / 5 | 0.0064 |
| CLR + logistic regression (microbiome-ml) | 20 | 0.692 | 3.90 | 2 | +0.000 | 0 / 0 | 0.0353 |
| CLR + XGBoost | 20 | 0.683 | 4.25 | 2 | -0.009 | 8 / 12 | 0.604 |
| CLR + random forest (scikit-learn) | 20 | 0.667 | 4.78 | 2 | -0.025 | 7 / 13 | 0.158 |
| Presence/absence + random forest | 20 | 0.670 | 5.10 | 3 | -0.022 | 6 / 14 | 1.17 |
| TSS + random forest | 20 | 0.676 | 5.12 | 0 | -0.016 | 7 / 13 | 1.32 |
| log-TSS + random forest | 20 | 0.676 | 5.15 | 0 | -0.016 | 8 / 12 | 1.26 |
| ILR (data-driven) + logistic regression* | 16 | 0.705 | n/a | n/a | -0.013 | 4 / 12 | 0.0391 |

Ranks are among the 8 pipelines that produced scores on every dataset. Friedman test over those 8 pipelines: p = 0.29. Paired Wilcoxon over dataset means, against CLR + LR (unadjusted): CLR + RF -0.005 (p = 0.65), CLR + XGBoost -0.009 (p = 0.41), log-TSS + RF -0.016 (p = 0.11), TSS + RF -0.016 (p = 0.10), presence/absence + RF -0.022 (p = 0.011), ILR -0.013 over 16 datasets (p = 0.030), scikit-learn CLR + LR +0.0002 (p = 0.46), scikit-learn CLR + RF -0.025 (p = 0.07). microbiome-ml CLR + RF vs scikit-learn CLR + RF: +0.020 on average (median +0.004; microbiome-ml higher on 13 of 20, p = 0.07), driven mostly by the imbalanced IBD tasks (25 vs 85) where the scikit-learn forest scored 0.56 to 0.59.

Main findings: no pipeline is reliably best (best pipeline per dataset: CLR + RF 8, CLR + LR 7, presence/absence + RF 3, CLR + XGBoost 2); CLR + logistic regression is within 0.02 of the best fixed pipeline on 11 of 20 datasets and never more than 0.07 below it; the microbiome-ml and scikit-learn logistic regressions agree to within 0.004 on every dataset; early CRC stages (MP, S0) and Obesity are close to chance (0.50 to 0.60), whereas Cirrhosis (0.88) and CRC-Nation AUS (0.86) are well separated. Fold SDs are 0.04 to 0.12, so per-dataset differences below about 0.03 should not be interpreted. p-values are unadjusted for multiple comparisons (the presence/absence result does not survive a Bonferroni correction over seven comparisons).


## Limitations

- The `xgboost` feature uses smelt-ml's from-scratch implementation, not the reference C++ library.
- Phylogenetic ILR uses topology only (no PhILR coordinate weights); polytomies are resolved arbitrarily.
- Permutation importance shifts sample totals when a single taxon is shuffled and splits credit among correlated taxa.
- Random forests use all features per split (smartcore's sqrt(p) default underperformed badly on sparse-signal tables in our simulations; cause undiagnosed).
- smartcore's ElasticNet scales poorly with taxa count (~71 s at n=400, p=2000) and failed on presence/absence input.

Gaurav Sablok \
gsablok@proton.me
