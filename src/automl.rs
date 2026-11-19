//! Automated search over (transform x model), optionally refined by a
//! hyperparameter grid, using cross-validation inside the training data only.
//!
//! Stage 1 scores every transform x model pair with default hyperparameters.
//! Stage 2 (when `grid != Off`) grids the hyperparameters of the `refine_top`
//! best pairs. The reported cross-validation score of the winner is optimistic
//! because it is the maximum over many candidates; use held-out data for honest
//! performance.

use crate::cv::{kfold, take};
use crate::data::Dataset;
use crate::error::{MicrobiomeError, Result};
use crate::importance::{permutation_importance, TaxonImportance};
use crate::metrics;
use crate::models::{ClassifierKind, FittedClassifier, FittedRegressor, RegressorKind};
use crate::params::{GridLevel, Params};
use crate::phylo::{PhyloTree, Phylogeny};
use crate::transform::{Preprocessor, PreprocessorConfig, Transform};
use std::fmt::{Debug, Write};

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone)]
pub struct Candidate<K> {
    pub transform: Transform,
    pub model: K,
    pub params: Params,
    /// Higher is better (balanced accuracy, or R2).
    pub score: f64,
    pub score_std: f64,
    /// Secondary metric: macro-F1 for classification, RMSE for regression.
    pub secondary: f64,
}

fn mean_std(v: &[f64]) -> (f64, f64) {
    let m = v.iter().sum::<f64>() / v.len().max(1) as f64;
    (
        m,
        (v.iter().map(|a| (a - m).powi(2)).sum::<f64>() / v.len().max(1) as f64).sqrt(),
    )
}

fn pre_cfg(base: &PreprocessorConfig, t: Transform, scale: bool) -> PreprocessorConfig {
    PreprocessorConfig {
        transform: t,
        standardize: scale || base.standardize,
        ..base.clone()
    }
}

fn sort_board<K>(board: &mut [Candidate<K>]) {
    board.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn report_table<K: Debug>(
    board: &[Candidate<K>],
    failures: &[String],
    top: usize,
    head: (&str, &str),
    selected: String,
) -> String {
    let mut s = format!(
        "transform         model               params                       {:<8}  (sd)    {}\n",
        head.0, head.1
    );
    for c in board.iter().take(top) {
        let _ = writeln!(
            s,
            "{:<17} {:<19} {:<28} {:.3}   ({:.3})  {:.3}",
            format!("{:?}", c.transform),
            format!("{:?}", c.model),
            c.params.to_string(),
            c.score,
            c.score_std,
            c.secondary
        );
    }
    if board.len() > top {
        let _ = writeln!(s, "... {} more candidates not shown", board.len() - top);
    }
    for f in failures {
        let _ = writeln!(s, "FAILED (excluded): {f}");
    }
    s.push_str(&selected);
    s
}

// ---------------------------------------------------------------- classification

#[derive(Debug, Clone)]
pub struct AutoClassifier {
    pub transforms: Vec<Transform>,
    pub models: Vec<ClassifierKind>,
    pub preprocess: PreprocessorConfig,
    pub cv_folds: usize,
    pub seed: u64,
    /// Hyperparameter grid for the best pairs (default `Off`).
    pub grid: GridLevel,
    /// How many of the best (transform, model) pairs get their grid searched.
    pub refine_top: usize,
}

impl Default for AutoClassifier {
    fn default() -> Self {
        Self {
            transforms: vec![
                Transform::Clr,
                Transform::LogTss,
                Transform::Tss,
                Transform::ArcsinSqrt,
                Transform::PresenceAbsence,
            ],
            models: ClassifierKind::all(),
            preprocess: PreprocessorConfig::default(),
            cv_folds: 5,
            seed: 42,
            grid: GridLevel::Off,
            refine_top: 3,
        }
    }
}

impl AutoClassifier {
    /// Attach a phylogeny and add the phylogenetic transforms to the search
    /// (`Ilr`, and `CladeClr` when the tree has branch lengths).
    pub fn with_phylogeny(mut self, tree: PhyloTree, feature_names: &[String]) -> Self {
        let has_len = tree.has_branch_lengths();
        self.preprocess.phylogeny = Some(Phylogeny::new(tree, feature_names));
        for t in [Transform::Ilr]
            .into_iter()
            .chain(has_len.then_some(Transform::CladeClr))
        {
            if !self.transforms.contains(&t) {
                self.transforms.push(t);
            }
        }
        self
    }
    pub fn with_grid(mut self, level: GridLevel) -> Self {
        self.grid = level;
        self
    }
}

pub struct FittedAutoClassifier {
    pub classes: Vec<String>,
    pub best: Candidate<ClassifierKind>,
    pub leaderboard: Vec<Candidate<ClassifierKind>>,
    pub preprocessor: Preprocessor,
    pub model: FittedClassifier,
    pub feature_names: Vec<String>,
    /// Candidates that could not be evaluated, with the reason.
    pub failures: Vec<String>,
}

impl AutoClassifier {
    pub fn fit(&self, data: &Dataset, labels: &[String]) -> Result<FittedAutoClassifier> {
        if labels.len() != data.n_samples() {
            return Err(MicrobiomeError::InvalidInput(
                "labels length != samples".into(),
            ));
        }
        let mut classes: Vec<String> = labels.to_vec();
        classes.sort();
        classes.dedup();
        let nc = classes.len();
        let y: Vec<u32> = labels
            .iter()
            .map(|l| classes.iter().position(|c| c == l).unwrap() as u32)
            .collect();
        let min_class = (0..nc)
            .map(|c| y.iter().filter(|&&v| v as usize == c).count())
            .min()
            .unwrap_or(0);
        if min_class < 2 {
            return Err(MicrobiomeError::InvalidInput(
                "every class needs at least 2 samples".into(),
            ));
        }
        let folds = kfold(y.len(), self.cv_folds.min(min_class), Some(&y), self.seed);

        let eval = |t: Transform,
                    m: ClassifierKind,
                    params: &Params|
         -> std::result::Result<(f64, f64, f64), String> {
            let (mut bal, mut f1s) = (vec![], vec![]);
            for (tr, te) in &folds {
                let run = || -> Result<(f64, f64)> {
                    let (xtr, xte) = (take(&data.x, tr), take(&data.x, te));
                    let (ytr, yte) = (take(&y, tr), take(&y, te));
                    let mut pre =
                        Preprocessor::new(pre_cfg(&self.preprocess, t, m.wants_scaling()));
                    let ztr = pre.fit_transform(&xtr)?;
                    let zte = pre.transform(&xte)?;
                    let pred = FittedClassifier::fit_with(m, params, &ztr, &ytr, nc, self.seed)?
                        .predict(&zte)?;
                    Ok((
                        metrics::balanced_accuracy(&yte, &pred, nc),
                        metrics::macro_f1(&yte, &pred, nc),
                    ))
                };
                match run() {
                    Ok((b, f)) => {
                        bal.push(b);
                        f1s.push(f);
                    }
                    Err(e) => return Err(format!("{t:?} + {m:?} [{params}]: {e}")),
                }
            }
            let (s, sd) = mean_std(&bal);
            Ok((s, sd, mean_std(&f1s).0))
        };

        let mut board: Vec<Candidate<ClassifierKind>> = Vec::new();
        let mut failures = Vec::new();
        let mut push = |board: &mut Vec<Candidate<ClassifierKind>>, t, m, params: Params| match eval(
            t, m, &params,
        ) {
            Ok((score, score_std, secondary)) => board.push(Candidate {
                transform: t,
                model: m,
                params,
                score,
                score_std,
                secondary,
            }),
            Err(e) => failures.push(e),
        };
        for &t in &self.transforms {
            for &m in &self.models {
                push(&mut board, t, m, Params::new());
            }
        }
        sort_board(&mut board);
        if self.grid != GridLevel::Off {
            let top: Vec<_> = board
                .iter()
                .take(self.refine_top)
                .map(|c| (c.transform, c.model))
                .collect();
            for (t, m) in top {
                for params in m.grid(self.grid) {
                    push(&mut board, t, m, params);
                }
            }
            sort_board(&mut board);
        }
        let best = board.first().cloned().ok_or_else(|| {
            MicrobiomeError::Model(format!(
                "no candidate pipeline could be evaluated: {}",
                failures.join("; ")
            ))
        })?;

        let mut pre = Preprocessor::new(pre_cfg(
            &self.preprocess,
            best.transform,
            best.model.wants_scaling(),
        ));
        let z = pre.fit_transform(&data.x)?;
        let model = FittedClassifier::fit_with(best.model, &best.params, &z, &y, nc, self.seed)?;
        Ok(FittedAutoClassifier {
            classes,
            best,
            leaderboard: board,
            preprocessor: pre,
            model,
            feature_names: data.feature_names.clone(),
            failures,
        })
    }
}

impl FittedAutoClassifier {
    pub fn predict_indices(&self, x: &[Vec<f64>]) -> Result<Vec<u32>> {
        self.model.predict(&self.preprocessor.transform(x)?)
    }
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<String>> {
        Ok(self
            .predict_indices(x)?
            .into_iter()
            .map(|i| self.classes[i as usize].clone())
            .collect())
    }
    pub fn report(&self) -> String {
        self.report_top(20)
    }
    pub fn report_top(&self, top: usize) -> String {
        let sel = format!(
            "selected: {:?} + {:?} [{}]; {} taxa kept of {}",
            self.best.transform,
            self.best.model,
            self.best.params,
            self.preprocessor.kept.len(),
            self.feature_names.len()
        );
        report_table(
            &self.leaderboard,
            &self.failures,
            top,
            ("bal.acc", "macroF1"),
            sel,
        )
    }

    /// Permutation importance of each retained taxon, balanced-accuracy drop, on
    /// data that should be held out from training. Sorted most important first.
    pub fn taxon_importance(
        &self,
        x: &[Vec<f64>],
        labels: &[String],
        n_repeats: usize,
        seed: u64,
    ) -> Result<Vec<TaxonImportance>> {
        let y: Vec<u32> = labels
            .iter()
            .map(|l| {
                self.classes
                    .iter()
                    .position(|c| c == l)
                    .map(|i| i as u32)
                    .ok_or_else(|| MicrobiomeError::InvalidInput(format!("unknown class '{l}'")))
            })
            .collect::<Result<_>>()?;
        let nc = self.classes.len();
        let score = |t: &[Vec<f64>]| -> Result<f64> {
            Ok(metrics::balanced_accuracy(
                &y,
                &self.predict_indices(t)?,
                nc,
            ))
        };
        permutation_importance(
            x,
            &self.feature_names,
            &self.preprocessor.kept,
            score,
            n_repeats,
            seed,
        )
    }

    /// Model-native importances labelled by the *output* feature (taxon, balance
    /// or clade), most important first. Only the XGBoost backend provides them.
    pub fn native_importance(&self) -> Option<Vec<(String, f64)>> {
        let imp = self.model.native_importance()?;
        let mut v: Vec<(String, f64)> = self
            .preprocessor
            .output_names(&self.feature_names)
            .into_iter()
            .zip(imp)
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(v)
    }
}

// ---------------------------------------------------------------- regression

#[derive(Debug, Clone)]
pub struct AutoRegressor {
    pub transforms: Vec<Transform>,
    pub models: Vec<RegressorKind>,
    pub preprocess: PreprocessorConfig,
    pub cv_folds: usize,
    pub seed: u64,
    pub grid: GridLevel,
    pub refine_top: usize,
}

impl Default for AutoRegressor {
    fn default() -> Self {
        Self {
            transforms: vec![
                Transform::Clr,
                Transform::LogTss,
                Transform::Tss,
                Transform::ArcsinSqrt,
                Transform::PresenceAbsence,
            ],
            models: RegressorKind::all(),
            preprocess: PreprocessorConfig::default(),
            cv_folds: 5,
            seed: 42,
            grid: GridLevel::Off,
            refine_top: 3,
        }
    }
}

impl AutoRegressor {
    pub fn with_phylogeny(mut self, tree: PhyloTree, feature_names: &[String]) -> Self {
        let has_len = tree.has_branch_lengths();
        self.preprocess.phylogeny = Some(Phylogeny::new(tree, feature_names));
        for t in [Transform::Ilr]
            .into_iter()
            .chain(has_len.then_some(Transform::CladeClr))
        {
            if !self.transforms.contains(&t) {
                self.transforms.push(t);
            }
        }
        self
    }
    pub fn with_grid(mut self, level: GridLevel) -> Self {
        self.grid = level;
        self
    }
}

pub struct FittedAutoRegressor {
    pub best: Candidate<RegressorKind>,
    pub leaderboard: Vec<Candidate<RegressorKind>>,
    pub preprocessor: Preprocessor,
    pub model: FittedRegressor,
    pub feature_names: Vec<String>,
    pub failures: Vec<String>,
}

impl AutoRegressor {
    pub fn fit(&self, data: &Dataset, y: &[f64]) -> Result<FittedAutoRegressor> {
        if y.len() != data.n_samples() || y.iter().any(|v| !v.is_finite()) {
            return Err(MicrobiomeError::InvalidInput(
                "targets must be finite and match sample count".into(),
            ));
        }
        if y.len() < 6 {
            return Err(MicrobiomeError::InvalidInput(
                "need at least 6 samples".into(),
            ));
        }
        let folds = kfold(y.len(), self.cv_folds, None, self.seed);

        let eval = |t: Transform,
                    m: RegressorKind,
                    params: &Params|
         -> std::result::Result<(f64, f64, f64), String> {
            let (mut r2s, mut rm) = (vec![], vec![]);
            for (tr, te) in &folds {
                let run = || -> Result<(f64, f64)> {
                    let (xtr, xte) = (take(&data.x, tr), take(&data.x, te));
                    let (ytr, yte) = (take(y, tr), take(y, te));
                    let mut pre =
                        Preprocessor::new(pre_cfg(&self.preprocess, t, m.wants_scaling()));
                    let ztr = pre.fit_transform(&xtr)?;
                    let zte = pre.transform(&xte)?;
                    let pred = FittedRegressor::fit_with(m, params, &ztr, &ytr, self.seed)?
                        .predict(&zte)?;
                    Ok((metrics::r2(&yte, &pred), metrics::rmse(&yte, &pred)))
                };
                match run() {
                    Ok((a, b)) => {
                        r2s.push(a);
                        rm.push(b);
                    }
                    Err(e) => return Err(format!("{t:?} + {m:?} [{params}]: {e}")),
                }
            }
            let (s, sd) = mean_std(&r2s);
            Ok((s, sd, mean_std(&rm).0))
        };

        let mut board: Vec<Candidate<RegressorKind>> = Vec::new();
        let mut failures = Vec::new();
        let mut push = |board: &mut Vec<Candidate<RegressorKind>>, t, m, params: Params| match eval(
            t, m, &params,
        ) {
            Ok((score, score_std, secondary)) => board.push(Candidate {
                transform: t,
                model: m,
                params,
                score,
                score_std,
                secondary,
            }),
            Err(e) => failures.push(e),
        };
        for &t in &self.transforms {
            for &m in &self.models {
                push(&mut board, t, m, Params::new());
            }
        }
        sort_board(&mut board);
        if self.grid != GridLevel::Off {
            let top: Vec<_> = board
                .iter()
                .take(self.refine_top)
                .map(|c| (c.transform, c.model))
                .collect();
            for (t, m) in top {
                for params in m.grid(self.grid) {
                    push(&mut board, t, m, params);
                }
            }
            sort_board(&mut board);
        }
        let best = board.first().cloned().ok_or_else(|| {
            MicrobiomeError::Model(format!(
                "no candidate pipeline could be evaluated: {}",
                failures.join("; ")
            ))
        })?;
        let mut pre = Preprocessor::new(pre_cfg(
            &self.preprocess,
            best.transform,
            best.model.wants_scaling(),
        ));
        let z = pre.fit_transform(&data.x)?;
        let model = FittedRegressor::fit_with(best.model, &best.params, &z, y, self.seed)?;
        Ok(FittedAutoRegressor {
            best,
            leaderboard: board,
            preprocessor: pre,
            model,
            feature_names: data.feature_names.clone(),
            failures,
        })
    }
}

impl FittedAutoRegressor {
    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        self.model.predict(&self.preprocessor.transform(x)?)
    }
    pub fn report(&self) -> String {
        self.report_top(20)
    }
    pub fn report_top(&self, top: usize) -> String {
        let sel = format!(
            "selected: {:?} + {:?} [{}]; {} taxa kept of {}",
            self.best.transform,
            self.best.model,
            self.best.params,
            self.preprocessor.kept.len(),
            self.feature_names.len()
        );
        report_table(&self.leaderboard, &self.failures, top, ("R2", "RMSE"), sel)
    }

    /// Permutation importance of each retained taxon, R2 drop, on held-out data.
    pub fn taxon_importance(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        n_repeats: usize,
        seed: u64,
    ) -> Result<Vec<TaxonImportance>> {
        let score = |t: &[Vec<f64>]| -> Result<f64> { Ok(metrics::r2(y, &self.predict(t)?)) };
        permutation_importance(
            x,
            &self.feature_names,
            &self.preprocessor.kept,
            score,
            n_repeats,
            seed,
        )
    }

    pub fn native_importance(&self) -> Option<Vec<(String, f64)>> {
        let imp = self.model.native_importance()?;
        let mut v: Vec<(String, f64)> = self
            .preprocessor
            .output_names(&self.feature_names)
            .into_iter()
            .zip(imp)
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Some(v)
    }
}
