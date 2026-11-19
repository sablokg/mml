//! Uniform wrappers over smartcore (and, with the `xgboost` feature, smelt-ml
//! XGBoost) estimators, with named hyperparameters and search grids.

use crate::error::{model_err, MicrobiomeError, Result};
use crate::params::{cross, GridLevel, Params};
use crate::ridge::Ridge;
use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::elastic_net::{ElasticNet, ElasticNetParameters};
use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::neighbors::knn_classifier::{KNNClassifier, KNNClassifierParameters};
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};
use smartcore::tree::decision_tree_regressor::{
    DecisionTreeRegressor, DecisionTreeRegressorParameters,
};

/*
Gaurav Sablok
gsablok@proton.me
*/

type M = DenseMatrix<f64>;

pub(crate) fn to_matrix(x: &[Vec<f64>]) -> Result<M> {
    Ok(M::from_2d_vec(&x.to_vec()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClassifierKind {
    RandomForest,
    LogisticRegression,
    DecisionTree,
    Knn,
    #[cfg(feature = "xgboost")]
    XgBoost,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegressorKind {
    RandomForest,
    ElasticNet,
    Ridge,
    DecisionTree,
    #[cfg(feature = "xgboost")]
    XgBoost,
}

const MTRY_SMALL: &[f64] = &[0.1, 0.33, 1.0];
const MTRY_FULL: &[f64] = &[0.05, 0.1, 0.33, 0.6, 1.0];
const LEAF_SMALL: &[f64] = &[1.0, 3.0];
const LEAF_FULL: &[f64] = &[1.0, 2.0, 5.0];

fn rf_grid(level: GridLevel) -> Vec<Params> {
    match level {
        GridLevel::Off => vec![],
        GridLevel::Small => cross(&[("mtry", MTRY_SMALL), ("min_leaf", LEAF_SMALL)]),
        GridLevel::Full => cross(&[("mtry", MTRY_FULL), ("min_leaf", LEAF_FULL)]),
    }
}

fn tree_grid(level: GridLevel) -> Vec<Params> {
    match level {
        GridLevel::Off => vec![],
        GridLevel::Small => cross(&[("max_depth", &[4.0, 8.0]), ("min_leaf", &[1.0, 5.0])]),
        GridLevel::Full => cross(&[
            ("max_depth", &[3.0, 5.0, 8.0, 12.0]),
            ("min_leaf", &[1.0, 3.0, 5.0]),
        ]),
    }
}

#[cfg(feature = "xgboost")]
fn xgb_grid(level: GridLevel) -> Vec<Params> {
    match level {
        GridLevel::Off => vec![],
        GridLevel::Small => cross(&[
            ("n_estimators", &[100.0, 300.0]),
            ("max_depth", &[3.0, 5.0]),
            ("lr", &[0.1]),
        ]),
        GridLevel::Full => cross(&[
            ("n_estimators", &[100.0, 300.0]),
            ("max_depth", &[2.0, 3.0, 5.0, 7.0]),
            ("lr", &[0.05, 0.1, 0.3]),
        ]),
    }
}

impl ClassifierKind {
    pub fn all() -> Vec<Self> {
        vec![
            Self::RandomForest,
            Self::LogisticRegression,
            Self::DecisionTree,
            Self::Knn,
            #[cfg(feature = "xgboost")]
            Self::XgBoost,
        ]
    }
    /// Models that behave best on standardised features.
    pub fn wants_scaling(&self) -> bool {
        matches!(self, Self::LogisticRegression | Self::Knn)
    }
    /// Hyperparameter grid for this model (empty when `level` is `Off`).
    pub fn grid(&self, level: GridLevel) -> Vec<Params> {
        match (self, level) {
            (_, GridLevel::Off) => vec![],
            (Self::RandomForest, l) => rf_grid(l),
            (Self::LogisticRegression, GridLevel::Small) => cross(&[("alpha", &[0.1, 1.0, 10.0])]),
            (Self::LogisticRegression, GridLevel::Full) => {
                cross(&[("alpha", &[0.001, 0.01, 0.1, 1.0, 10.0, 100.0])])
            }
            (Self::DecisionTree, l) => tree_grid(l),
            (Self::Knn, GridLevel::Small) => cross(&[("k", &[3.0, 7.0, 15.0])]),
            (Self::Knn, GridLevel::Full) => cross(&[("k", &[3.0, 5.0, 9.0, 15.0, 25.0])]),
            #[cfg(feature = "xgboost")]
            (Self::XgBoost, l) => xgb_grid(l),
        }
    }
}

impl RegressorKind {
    pub fn all() -> Vec<Self> {
        vec![
            Self::RandomForest,
            Self::ElasticNet,
            Self::Ridge,
            Self::DecisionTree,
            #[cfg(feature = "xgboost")]
            Self::XgBoost,
        ]
    }
    pub fn wants_scaling(&self) -> bool {
        matches!(self, Self::ElasticNet | Self::Ridge)
    }
    pub fn grid(&self, level: GridLevel) -> Vec<Params> {
        match (self, level) {
            (_, GridLevel::Off) => vec![],
            (Self::RandomForest, l) => rf_grid(l),
            (Self::ElasticNet, GridLevel::Small) => {
                cross(&[("alpha", &[0.01, 0.1, 1.0]), ("l1", &[0.5])])
            }
            (Self::ElasticNet, GridLevel::Full) => cross(&[
                ("alpha", &[0.003, 0.01, 0.03, 0.1, 0.3, 1.0]),
                ("l1", &[0.1, 0.5, 0.9]),
            ]),
            // Ridge selects its own penalty by leave-one-out; nothing to grid.
            (Self::Ridge, _) => vec![],
            (Self::DecisionTree, l) => tree_grid(l),
            #[cfg(feature = "xgboost")]
            (Self::XgBoost, l) => xgb_grid(l),
        }
    }
}

fn mtry(params: &Params, n_features: usize) -> usize {
    ((params.get_or("mtry", 1.0) * n_features as f64).round() as usize).clamp(1, n_features.max(1))
}

pub enum FittedClassifier {
    Rf(RandomForestClassifier<f64, u32, M, Vec<u32>>),
    Logit(LogisticRegression<f64, u32, M, Vec<u32>>),
    Tree(DecisionTreeClassifier<f64, u32, M, Vec<u32>>),
    Knn(
        KNNClassifier<
            f64,
            u32,
            M,
            Vec<u32>,
            smartcore::metrics::distance::euclidian::Euclidian<f64>,
        >,
    ),
    #[cfg(feature = "xgboost")]
    Xgb(crate::xgb::XgbModel),
}

impl FittedClassifier {
    /// Fit with default hyperparameters. `y` must be class indices `0..n_classes`.
    pub fn fit(
        kind: ClassifierKind,
        x: &[Vec<f64>],
        y: &[u32],
        n_classes: usize,
        seed: u64,
    ) -> Result<Self> {
        Self::fit_with(kind, &Params::new(), x, y, n_classes, seed)
    }

    pub fn fit_with(
        kind: ClassifierKind,
        params: &Params,
        x: &[Vec<f64>],
        y: &[u32],
        n_classes: usize,
        seed: u64,
    ) -> Result<Self> {
        if n_classes < 2 {
            return Err(MicrobiomeError::InvalidInput(
                "need at least two classes".into(),
            ));
        }
        let xm = to_matrix(x)?;
        let yv = y.to_vec();
        let p = x.first().map(|r| r.len()).unwrap_or(0);
        Ok(match kind {
            ClassifierKind::RandomForest => {
                let mut pr = RandomForestClassifierParameters::default();
                pr.n_trees = params.get_or("n_trees", 300.0) as _;
                pr.seed = seed;
                // All features per split by default: smartcore's sqrt(p) default performs
                // poorly on sparse-signal microbiome tables (see benchmarks).
                pr.m = Some(mtry(params, p));
                pr.min_samples_leaf = params.get_or("min_leaf", 1.0) as _;
                if let Some(d) = params.get("max_depth") {
                    pr.max_depth = Some(d as _);
                }
                Self::Rf(RandomForestClassifier::fit(&xm, &yv, pr).map_err(model_err)?)
            }
            ClassifierKind::LogisticRegression => {
                let pr =
                    LogisticRegressionParameters::default().with_alpha(params.get_or("alpha", 1.0));
                Self::Logit(LogisticRegression::fit(&xm, &yv, pr).map_err(model_err)?)
            }
            ClassifierKind::DecisionTree => {
                let mut pr = DecisionTreeClassifierParameters::default();
                pr.max_depth = Some(params.get_or("max_depth", 8.0) as _);
                pr.min_samples_leaf = params.get_or("min_leaf", 1.0) as _;
                pr.seed = Some(seed);
                Self::Tree(DecisionTreeClassifier::fit(&xm, &yv, pr).map_err(model_err)?)
            }
            ClassifierKind::Knn => {
                let default_k = ((x.len() as f64).sqrt() as usize).clamp(3, 15) | 1;
                let k = (params.get_or("k", default_k as f64) as usize).clamp(1, x.len());
                Self::Knn(
                    KNNClassifier::fit(&xm, &yv, KNNClassifierParameters::default().with_k(k))
                        .map_err(model_err)?,
                )
            }
            #[cfg(feature = "xgboost")]
            ClassifierKind::XgBoost => Self::Xgb(crate::xgb::XgbModel::fit_classifier(
                x, y, n_classes, params, seed,
            )?),
        })
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<u32>> {
        let xm = to_matrix(x)?;
        match self {
            Self::Rf(m) => m.predict(&xm).map_err(model_err),
            Self::Logit(m) => m.predict(&xm).map_err(model_err),
            Self::Tree(m) => m.predict(&xm).map_err(model_err),
            Self::Knn(m) => m.predict(&xm).map_err(model_err),
            #[cfg(feature = "xgboost")]
            Self::Xgb(m) => m.predict_classes(x),
        }
    }

    /// Model-native importances, normalised to sum 1, in input-column order.
    /// Only the XGBoost backend provides them; use permutation importance otherwise.
    pub fn native_importance(&self) -> Option<Vec<f64>> {
        match self {
            #[cfg(feature = "xgboost")]
            Self::Xgb(m) => m.native_importance(),
            _ => None,
        }
    }
}

pub enum FittedRegressor {
    Rf(RandomForestRegressor<f64, f64, M, Vec<f64>>),
    Enet(ElasticNet<f64, f64, M, Vec<f64>>),
    Ridge(Ridge),
    Tree(DecisionTreeRegressor<f64, f64, M, Vec<f64>>),
    #[cfg(feature = "xgboost")]
    Xgb(crate::xgb::XgbModel),
}

impl FittedRegressor {
    pub fn fit(kind: RegressorKind, x: &[Vec<f64>], y: &[f64], seed: u64) -> Result<Self> {
        Self::fit_with(kind, &Params::new(), x, y, seed)
    }

    pub fn fit_with(
        kind: RegressorKind,
        params: &Params,
        x: &[Vec<f64>],
        y: &[f64],
        seed: u64,
    ) -> Result<Self> {
        let xm = to_matrix(x)?;
        let yv = y.to_vec();
        let p = x.first().map(|r| r.len()).unwrap_or(0);
        Ok(match kind {
            RegressorKind::RandomForest => {
                let mut pr = RandomForestRegressorParameters::default();
                pr.n_trees = params.get_or("n_trees", 300.0) as _;
                pr.seed = seed;
                pr.m = Some(mtry(params, p));
                pr.min_samples_leaf = params.get_or("min_leaf", 1.0) as _;
                if let Some(d) = params.get("max_depth") {
                    pr.max_depth = Some(d as _);
                }
                Self::Rf(RandomForestRegressor::fit(&xm, &yv, pr).map_err(model_err)?)
            }
            RegressorKind::ElasticNet => {
                let pr = ElasticNetParameters::default()
                    .with_alpha(params.get_or("alpha", 0.1))
                    .with_l1_ratio(params.get_or("l1", 0.5));
                Self::Enet(ElasticNet::fit(&xm, &yv, pr).map_err(model_err)?)
            }
            RegressorKind::Ridge => Self::Ridge(Ridge::fit(x, y)?),
            RegressorKind::DecisionTree => {
                let mut pr = DecisionTreeRegressorParameters::default();
                pr.max_depth = Some(params.get_or("max_depth", 8.0) as _);
                pr.min_samples_leaf = params.get_or("min_leaf", 1.0) as _;
                pr.seed = Some(seed);
                Self::Tree(DecisionTreeRegressor::fit(&xm, &yv, pr).map_err(model_err)?)
            }
            #[cfg(feature = "xgboost")]
            RegressorKind::XgBoost => {
                Self::Xgb(crate::xgb::XgbModel::fit_regressor(x, y, params, seed)?)
            }
        })
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        let xm = to_matrix(x)?;
        match self {
            Self::Rf(m) => m.predict(&xm).map_err(model_err),
            Self::Enet(m) => m.predict(&xm).map_err(model_err),
            Self::Ridge(m) => Ok(m.predict(x)),
            Self::Tree(m) => m.predict(&xm).map_err(model_err),
            #[cfg(feature = "xgboost")]
            Self::Xgb(m) => m.predict_values(x),
        }
    }

    pub fn native_importance(&self) -> Option<Vec<f64>> {
        match self {
            #[cfg(feature = "xgboost")]
            Self::Xgb(m) => m.native_importance(),
            _ => None,
        }
    }
}
