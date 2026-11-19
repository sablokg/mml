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

### Real data: gut metagenomes in schizophrenia

Benchmark on the cohort of [Zhu et al. 2020, *Nature Communications* 11:1612](https://www.nature.com/articles/s41467-020-15457-9) (MetaPhlAn counts, 90 schizophrenia cases vs 81 healthy controls, n = 171), via the ML-for-microbiomics tutorial data. Repeated stratified 5-fold CV (3 repeats = 15 outer folds); all preprocessing is fitted inside each training fold, and AUTO's selection is nested in the training fold. Chance is 0.5. Code, data and results are in [`real_benchmark/`](real_benchmark/) (`./real_benchmark/run_all.sh`).

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

## Limitations

- The `xgboost` feature uses smelt-ml's from-scratch implementation, not the reference C++ library.
- Phylogenetic ILR uses topology only (no PhILR coordinate weights); polytomies are resolved arbitrarily.
- Permutation importance shifts sample totals when a single taxon is shuffled and splits credit among correlated taxa.
- Random forests use all features per split (smartcore's sqrt(p) default underperformed badly on sparse-signal tables in our simulations; cause undiagnosed).
- smartcore's ElasticNet scales poorly with taxa count (~71 s at n=400, p=2000) and failed on presence/absence input.

Gaurav Sablok \
gsablok@proton.me
