use crate::error::{MicrobiomeError, Result};
use std::path::Path;

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Samples x taxa abundance table (counts or relative abundances).
#[derive(Debug, Clone)]
pub struct Dataset {
    pub x: Vec<Vec<f64>>,
    pub feature_names: Vec<String>,
    pub sample_ids: Vec<String>,
}

impl Dataset {
    pub fn new(
        x: Vec<Vec<f64>>,
        feature_names: Option<Vec<String>>,
        sample_ids: Option<Vec<String>>,
    ) -> Result<Self> {
        let p = x.first().map(|r| r.len()).unwrap_or(0);
        for (i, r) in x.iter().enumerate() {
            if r.len() != p {
                return Err(MicrobiomeError::InvalidInput(format!(
                    "row {i} has {} columns, expected {p}",
                    r.len()
                )));
            }
            if r.iter().any(|v| !v.is_finite() || *v < 0.0) {
                return Err(MicrobiomeError::InvalidInput(format!(
                    "row {i} has negative or non-finite abundance"
                )));
            }
        }
        let feature_names =
            feature_names.unwrap_or_else(|| (0..p).map(|j| format!("taxon_{j}")).collect());
        let sample_ids =
            sample_ids.unwrap_or_else(|| (0..x.len()).map(|i| format!("sample_{i}")).collect());
        if feature_names.len() != p || sample_ids.len() != x.len() {
            return Err(MicrobiomeError::InvalidInput(
                "name lengths do not match table shape".into(),
            ));
        }
        Ok(Self {
            x,
            feature_names,
            sample_ids,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.x.len()
    }
    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Read a delimited table: header row of taxa names, first column = sample id.
    /// Set `taxa_in_rows` for the BIOM/QIIME style (taxa as rows, samples as columns).
    pub fn from_delimited<P: AsRef<Path>>(
        path: P,
        delim: char,
        taxa_in_rows: bool,
    ) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut lines = text
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.starts_with("# Constructed"));
        let header: Vec<&str> = lines
            .next()
            .ok_or_else(|| MicrobiomeError::InvalidInput("empty file".into()))?
            .split(delim)
            .collect();
        let mut names = Vec::new();
        let mut rows = Vec::new();
        for l in lines {
            let mut it = l.split(delim);
            names.push(it.next().unwrap_or("").trim().to_string());
            let row: std::result::Result<Vec<f64>, _> =
                it.map(|v| v.trim().parse::<f64>()).collect();
            rows.push(row.map_err(|e| MicrobiomeError::InvalidInput(format!("parse error: {e}")))?);
        }
        let cols: Vec<String> = header
            .iter()
            .skip(1)
            .map(|s| s.trim().to_string())
            .collect();
        if !taxa_in_rows {
            return Dataset::new(rows, Some(cols), Some(names));
        }
        let (p, n) = (rows.len(), cols.len());
        let mut x = vec![vec![0.0; p]; n];
        for (j, r) in rows.iter().enumerate() {
            if r.len() != n {
                return Err(MicrobiomeError::InvalidInput("ragged table".into()));
            }
            for (i, v) in r.iter().enumerate() {
                x[i][j] = *v;
            }
        }
        Dataset::new(x, Some(names), Some(cols))
    }
}
