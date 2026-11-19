//! Taxon-level permutation importance through the *whole* fitted pipeline.
//!
//! One raw taxon column is shuffled across samples, the table is pushed through
//! the fitted preprocessing (filter, normalise, CLR/ILR/clade aggregation, scale)
//! and model, and the drop in the score is recorded. Because the permutation is
//! applied to raw taxa, importances map back to taxon names even when the model
//! sees balances or clades. Taxa removed by filtering have no effect and are omitted.
//!
//! Caveats: shuffling one taxon also changes the sample's total, so relative
//! abundances of all taxa shift slightly; correlated taxa share importance; and
//! importance is only meaningful on data the model was *not* fitted to.

use crate::cv::Lcg;
use crate::error::Result;
use std::fmt::Write;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone)]
pub struct TaxonImportance {
    pub taxon: String,
    /// Column index in the original table.
    pub column: usize,
    /// Mean drop in score when this taxon is permuted (larger = more important).
    pub mean_drop: f64,
    pub std_drop: f64,
}

/// `score` maps a raw table to a performance value where higher is better.
pub fn permutation_importance<F>(
    x: &[Vec<f64>],
    names: &[String],
    cols: &[usize],
    mut score: F,
    n_repeats: usize,
    seed: u64,
) -> Result<Vec<TaxonImportance>>
where
    F: FnMut(&[Vec<f64>]) -> Result<f64>,
{
    let base = score(x)?;
    let n = x.len();
    let reps = n_repeats.max(1);
    let mut work = x.to_vec();
    let mut rng = Lcg::new(seed);
    let mut out = Vec::with_capacity(cols.len());
    for &c in cols {
        let orig: Vec<f64> = x.iter().map(|r| r[c]).collect();
        let mut drops = Vec::with_capacity(reps);
        for _ in 0..reps {
            let mut perm = orig.clone();
            rng.shuffle(&mut perm);
            for i in 0..n {
                work[i][c] = perm[i];
            }
            drops.push(base - score(&work)?);
        }
        for i in 0..n {
            work[i][c] = orig[i];
        }
        let m = drops.iter().sum::<f64>() / reps as f64;
        let sd = (drops.iter().map(|d| (d - m).powi(2)).sum::<f64>() / reps as f64).sqrt();
        out.push(TaxonImportance {
            taxon: names[c].clone(),
            column: c,
            mean_drop: m,
            std_drop: sd,
        });
    }
    out.sort_by(|a, b| {
        b.mean_drop
            .partial_cmp(&a.mean_drop)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(out)
}

/// Plain-text table of the `top` most important taxa.
pub fn format_importance(imp: &[TaxonImportance], top: usize) -> String {
    let mut s = String::from("rank  taxon                          score drop (sd)\n");
    for (i, t) in imp.iter().take(top).enumerate() {
        let _ = writeln!(
            s,
            "{:<5} {:<30} {:+.4} ({:.4})",
            i + 1,
            t.taxon,
            t.mean_drop,
            t.std_drop
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_informative_column_matters() {
        let x: Vec<Vec<f64>> = (0..40)
            .map(|i| vec![(i % 2) as f64, ((i * 7) % 5) as f64, ((i * 3) % 4) as f64])
            .collect();
        let y: Vec<f64> = (0..40).map(|i| (i % 2) as f64).collect();
        let names: Vec<String> = ["signal", "n1", "n2"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // "model": predicts column 0; score = fraction correct
        let score = |t: &[Vec<f64>]| {
            Ok(t.iter().zip(&y).filter(|(r, y)| r[0] == **y).count() as f64 / y.len() as f64)
        };
        let imp = permutation_importance(&x, &names, &[0, 1, 2], score, 10, 1).unwrap();
        assert_eq!(imp[0].taxon, "signal");
        assert!(imp[0].mean_drop > 0.2);
        assert!(imp[1].mean_drop.abs() < 1e-12 && imp[2].mean_drop.abs() < 1e-12);
    }
}
