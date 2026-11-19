//! Generic named hyperparameters and grid construction.

use std::collections::BTreeMap;
use std::fmt;

/*
Gaurav Sablok
gsablok@proton.me
*/

/// How exhaustive the hyperparameter grid is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridLevel {
    /// No tuning: every model uses its defaults.
    Off,
    /// A few values per hyperparameter.
    Small,
    /// A wider grid (slower).
    Full,
}

/// Named numeric hyperparameters. Missing keys mean "use the model default".
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Params(pub BTreeMap<String, f64>);

impl Params {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with(mut self, key: &str, v: f64) -> Self {
        self.0.insert(key.to_string(), v);
        self
    }
    pub fn get(&self, key: &str) -> Option<f64> {
        self.0.get(key).copied()
    }
    pub fn get_or(&self, key: &str, default: f64) -> f64 {
        self.get(key).unwrap_or(default)
    }
    pub fn is_default(&self) -> bool {
        self.0.is_empty()
    }
}

impl fmt::Display for Params {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return write!(f, "default");
        }
        let s: Vec<String> = self.0.iter().map(|(k, v)| format!("{k}={v}")).collect();
        write!(f, "{}", s.join(","))
    }
}

/// Cartesian product of named value lists.
pub fn cross(axes: &[(&str, &[f64])]) -> Vec<Params> {
    let mut out = vec![Params::new()];
    for (name, vals) in axes {
        let mut next = Vec::with_capacity(out.len() * vals.len());
        for p in &out {
            for v in *vals {
                next.push(p.clone().with(name, *v));
            }
        }
        out = next;
    }
    out
}
