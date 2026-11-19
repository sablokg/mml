//! # microbiome-ml
//!
//! Automated machine learning for microbiome count / abundance tables.
//!
//! * [`transform`]: prevalence filtering, total-sum scaling, CLR, log, arcsin-sqrt,
//!   presence/absence, ILR balances (phylogenetic or data-driven), clade-aggregated
//!   CLR and optional standardisation, fitted on training data only.
//! * [`phylo`]: Newick parsing, balances, clade grouping.
//! * [`importance`]: taxon-level permutation importance through the whole pipeline.
//! * [`models`]: classifiers and regressors backed by `smartcore` (and `xgboost`
//!   behind the `xgboost` cargo feature).
//! * [`cv`]: stratified / plain k-fold cross-validation and metrics.
//! * [`automl`]: searches transform x model combinations (and optionally a
//!   hyperparameter grid) by cross-validation and returns a fitted pipeline.
//!
//! ```no_run
//! use microbiome_ml::prelude::*;
//! let x: Vec<Vec<f64>> = vec![]; // samples x taxa counts
//! let y: Vec<String> = vec![];
//! let data = Dataset::new(x, None, None).unwrap();
//! let model = AutoClassifier::default().fit(&data, &y).unwrap();
//! println!("{}", model.report());
//! let preds = model.predict(&data.x).unwrap();
//! ```

pub mod automl;
pub mod cv;
pub mod data;
pub mod error;
pub mod importance;
pub mod metrics;
pub mod models;
pub mod params;
pub mod phylo;
pub mod ridge;
pub mod transform;
#[cfg(feature = "xgboost")]
pub mod xgb;

/*
Gaurav Sablok
gsablok@proton.me
*/

pub mod prelude {
    pub use crate::automl::{
        AutoClassifier, AutoRegressor, FittedAutoClassifier, FittedAutoRegressor,
    };
    pub use crate::data::Dataset;
    pub use crate::error::{MicrobiomeError, Result};
    pub use crate::importance::{format_importance, TaxonImportance};
    pub use crate::models::{ClassifierKind, RegressorKind};
    pub use crate::params::{GridLevel, Params};
    pub use crate::phylo::{PhyloTree, Phylogeny};
    pub use crate::transform::{Preprocessor, PreprocessorConfig, Transform};
}
