//! Pure-Rust XGBoost backend via the `smelt-ml` crate (feature `xgboost`).
//! Newton boosting with histogram splits; no C++ toolchain required.

use crate::error::{model_err, MicrobiomeError, Result};
use crate::params::Params;
use ndarray::Array2;
use smelt_ml::learner::{Learner, TrainedModel, XGBoost};
use smelt_ml::prediction::Prediction;
use smelt_ml::task::{ClassificationTask, RegressionTask};

/*
Gaurav Sablok
gsablok@proton.me
*/

pub struct XgbModel {
    model: Box<dyn TrainedModel>,
}

fn to_array(x: &[Vec<f64>]) -> Result<Array2<f64>> {
    let (n, p) = (x.len(), x.first().map(|r| r.len()).unwrap_or(0));
    Array2::from_shape_vec((n, p), x.iter().flatten().copied().collect()).map_err(model_err)
}

fn names(p: usize) -> Vec<String> {
    (0..p).map(|j| format!("f{j}")).collect()
}

fn learner(params: &Params, seed: u64) -> XGBoost {
    XGBoost::new()
        .with_n_estimators(params.get_or("n_estimators", 200.0) as usize)
        .with_max_depth(params.get_or("max_depth", 4.0) as usize)
        .with_learning_rate(params.get_or("lr", 0.1))
        .with_subsample(0.8)
        .with_colsample_bytree(0.8)
        .with_seed(seed)
}

impl XgbModel {
    pub fn fit_classifier(
        x: &[Vec<f64>],
        y: &[u32],
        _n_classes: usize,
        params: &Params,
        seed: u64,
    ) -> Result<Self> {
        let arr = to_array(x)?;
        let p = arr.ncols();
        let task =
            ClassificationTask::new("microbiome", arr, y.iter().map(|v| *v as usize).collect())
                .and_then(|t| t.with_feature_names(names(p)))
                .map_err(model_err)?;
        let model = learner(params, seed)
            .train_classif(&task)
            .map_err(model_err)?;
        Ok(Self { model })
    }

    pub fn fit_regressor(x: &[Vec<f64>], y: &[f64], params: &Params, seed: u64) -> Result<Self> {
        let arr = to_array(x)?;
        let p = arr.ncols();
        let task = RegressionTask::new("microbiome", arr, y.to_vec())
            .and_then(|t| t.with_feature_names(names(p)))
            .map_err(model_err)?;
        let model = learner(params, seed)
            .train_regress(&task)
            .map_err(model_err)?;
        Ok(Self { model })
    }

    pub fn predict_classes(&self, x: &[Vec<f64>]) -> Result<Vec<u32>> {
        match self.model.predict(&to_array(x)?).map_err(model_err)? {
            Prediction::Classification { predicted, .. } => {
                Ok(predicted.into_iter().map(|v| v as u32).collect())
            }
            _ => Err(MicrobiomeError::Model(
                "XGBoost returned a non-classification prediction".into(),
            )),
        }
    }

    pub fn predict_values(&self, x: &[Vec<f64>]) -> Result<Vec<f64>> {
        match self.model.predict(&to_array(x)?).map_err(model_err)? {
            Prediction::Regression { predicted, .. } => Ok(predicted),
            _ => Err(MicrobiomeError::Model(
                "XGBoost returned a non-regression prediction".into(),
            )),
        }
    }

    /// Gain-based importances in column order, summing to 1 (None if no splits).
    pub fn native_importance(&self) -> Option<Vec<f64>> {
        self.model
            .feature_importance()
            .map(|v| v.into_iter().map(|(_, i)| i).collect())
    }
}
