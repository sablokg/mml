//! Compositional preprocessing. Everything learned (kept taxa, pseudocount,
//! balance tree, clade groups, scaling) is fitted on training data only.

use crate::error::{MicrobiomeError, Result};
use crate::phylo::{bin_from_clade, clade_groups, dendrogram_from_logs, Balances, Phylogeny};
use std::collections::HashMap;

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Raw values after filtering, no transform.
    None,
    /// Total-sum scaling to relative abundance.
    Tss,
    /// log(TSS + pseudocount).
    LogTss,
    /// Arcsine square root of relative abundance.
    ArcsinSqrt,
    /// Centered log-ratio of TSS + pseudocount.
    Clr,
    /// 1 if taxon present, else 0.
    PresenceAbsence,
    /// Isometric log-ratio balances. With a phylogeny: balances at the tree's
    /// nodes (PhILR-style topology, no coordinate weighting). Without one: a
    /// data-driven dendrogram on the training variation matrix.
    Ilr,
    /// Sum taxa into maximal clades below a height cut of the phylogeny, then CLR.
    CladeClr,
}

impl Transform {
    /// Transforms that need no tree.
    pub const ALL: [Transform; 7] = [
        Transform::None,
        Transform::Tss,
        Transform::LogTss,
        Transform::ArcsinSqrt,
        Transform::Clr,
        Transform::PresenceAbsence,
        Transform::Ilr,
    ];
}

#[derive(Debug, Clone)]
pub struct PreprocessorConfig {
    pub transform: Transform,
    /// Keep taxa present (> 0) in at least this fraction of training samples.
    pub min_prevalence: f64,
    /// Keep taxa whose mean relative abundance is at least this.
    pub min_mean_abundance: f64,
    /// Pseudocount on relative abundance. `None` = half the smallest positive
    /// training relative abundance.
    pub pseudocount: Option<f64>,
    /// Z-score features after the transform (helps linear models and kNN).
    pub standardize: bool,
    /// Tree for `Ilr` (optional) and `CladeClr` (required).
    pub phylogeny: Option<Phylogeny>,
    /// `CladeClr` cut height as a fraction of the tree height (0 to 1).
    pub clade_height_fraction: f64,
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            transform: Transform::Clr,
            min_prevalence: 0.1,
            min_mean_abundance: 0.0,
            pseudocount: None,
            standardize: false,
            phylogeny: None,
            clade_height_fraction: 0.25,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Preprocessor {
    pub config: PreprocessorConfig,
    /// Original column indices of the retained taxa.
    pub kept: Vec<usize>,
    pub pseudocount: f64,
    /// Kept taxa dropped because they are absent from the phylogeny.
    pub unplaced: usize,
    balances: Option<Balances>,
    groups: Vec<Vec<usize>>,
    means: Vec<f64>,
    stds: Vec<f64>,
    fitted: bool,
}

/// Total-sum scaling of one sample over the given columns. All-zero rows stay zero.
fn tss_row(row: &[f64], cols: &[usize]) -> Vec<f64> {
    let total: f64 = cols.iter().map(|&j| row[j]).sum();
    if total > 0.0 {
        cols.iter().map(|&j| row[j] / total).collect()
    } else {
        vec![0.0; cols.len()]
    }
}

/// Centered log-ratio of a single composition (adds `pc` to every part).
pub fn clr(row: &[f64], pc: f64) -> Vec<f64> {
    let logs: Vec<f64> = row.iter().map(|v| (v + pc).ln()).collect();
    let m = logs.iter().sum::<f64>() / logs.len().max(1) as f64;
    logs.into_iter().map(|l| l - m).collect()
}

fn min_positive(rows: &[Vec<f64>]) -> f64 {
    rows.iter()
        .flatten()
        .copied()
        .filter(|v| *v > 0.0)
        .fold(f64::INFINITY, f64::min)
}

impl Preprocessor {
    pub fn new(config: PreprocessorConfig) -> Self {
        Self {
            config,
            kept: vec![],
            pseudocount: 0.0,
            unplaced: 0,
            balances: None,
            groups: vec![],
            means: vec![],
            stds: vec![],
            fitted: false,
        }
    }

    pub fn fit(&mut self, x: &[Vec<f64>]) -> Result<()> {
        if x.is_empty() {
            return Err(MicrobiomeError::InvalidInput("empty table".into()));
        }
        let (n, p) = (x.len(), x[0].len());
        let all: Vec<usize> = (0..p).collect();
        let rel_all: Vec<Vec<f64>> = x.iter().map(|r| tss_row(r, &all)).collect();

        self.kept = (0..p)
            .filter(|&j| {
                let prev = x.iter().filter(|r| r[j] > 0.0).count() as f64 / n as f64;
                let mean = rel_all.iter().map(|r| r[j]).sum::<f64>() / n as f64;
                prev >= self.config.min_prevalence && mean >= self.config.min_mean_abundance
            })
            .collect();

        let t = self.config.transform;
        let uses_tree =
            t == Transform::CladeClr || (t == Transform::Ilr && self.config.phylogeny.is_some());
        self.unplaced = 0;
        if uses_tree {
            let ph = self.config.phylogeny.as_ref().ok_or_else(|| {
                MicrobiomeError::InvalidInput("CladeClr needs a phylogeny".into())
            })?;
            if ph.names.len() != p {
                return Err(MicrobiomeError::InvalidInput(format!(
                    "phylogeny was built for {} columns, table has {p}",
                    ph.names.len()
                )));
            }
            let leaves: std::collections::HashSet<String> =
                ph.tree.leaf_names().into_iter().collect();
            let before = self.kept.len();
            self.kept.retain(|&j| leaves.contains(&ph.names[j]));
            self.unplaced = before - self.kept.len();
        }
        if self.kept.len() < 2 {
            return Err(MicrobiomeError::InvalidInput(format!(
                "only {} taxa pass filters; relax prevalence/abundance or check tree names",
                self.kept.len()
            )));
        }

        let rel_kept: Vec<Vec<f64>> = x.iter().map(|r| tss_row(r, &self.kept)).collect();
        let pc_of = |rows: &[Vec<f64>], cfg: Option<f64>| {
            cfg.unwrap_or_else(|| {
                let m = min_positive(rows);
                if m.is_finite() {
                    m / 2.0
                } else {
                    1e-6
                }
            })
        };
        self.pseudocount = pc_of(&rel_kept, self.config.pseudocount);
        self.balances = None;
        self.groups.clear();

        match t {
            Transform::Ilr => {
                let bin = if let Some(ph) = &self.config.phylogeny {
                    let pos: HashMap<&str, usize> = self
                        .kept
                        .iter()
                        .enumerate()
                        .map(|(k, &j)| (ph.names[j].as_str(), k))
                        .collect();
                    bin_from_clade(&ph.tree.root, &pos).ok_or_else(|| {
                        MicrobiomeError::InvalidInput("no kept taxa found in tree".into())
                    })?
                } else {
                    let logs: Vec<Vec<f64>> = rel_kept
                        .iter()
                        .map(|r| r.iter().map(|v| (v + self.pseudocount).ln()).collect())
                        .collect();
                    dendrogram_from_logs(&logs)?
                };
                self.balances = Some(Balances::from_bin(&bin));
            }
            Transform::CladeClr => {
                let ph = self.config.phylogeny.as_ref().unwrap();
                if !ph.tree.has_branch_lengths() {
                    return Err(MicrobiomeError::InvalidInput(
                        "CladeClr needs a tree with branch lengths; use Ilr instead".into(),
                    ));
                }
                let pos: HashMap<&str, usize> = self
                    .kept
                    .iter()
                    .enumerate()
                    .map(|(k, &j)| (ph.names[j].as_str(), k))
                    .collect();
                let cut = self.config.clade_height_fraction.clamp(0.0, 1.0) * ph.tree.height();
                let groups = clade_groups(&ph.tree.root, &pos, cut);
                if groups.len() < 2 {
                    return Err(MicrobiomeError::InvalidInput(
                        "clade cut leaves fewer than 2 groups; lower clade_height_fraction".into(),
                    ));
                }
                let agg: Vec<Vec<f64>> = rel_kept
                    .iter()
                    .map(|r| {
                        groups
                            .iter()
                            .map(|g| g.iter().map(|&k| r[k]).sum())
                            .collect()
                    })
                    .collect();
                self.pseudocount = pc_of(&agg, self.config.pseudocount);
                self.groups = groups;
            }
            _ => {}
        }

        self.fitted = true;
        self.means.clear();
        self.stds.clear();
        if self.config.standardize {
            let z: Vec<Vec<f64>> = x.iter().map(|r| self.apply(r)).collect();
            let k = z[0].len();
            for j in 0..k {
                let m = z.iter().map(|r| r[j]).sum::<f64>() / n as f64;
                let var = z.iter().map(|r| (r[j] - m).powi(2)).sum::<f64>() / n as f64;
                self.means.push(m);
                self.stds
                    .push(if var.sqrt() > 1e-12 { var.sqrt() } else { 1.0 });
            }
        }
        Ok(())
    }

    fn apply(&self, row: &[f64]) -> Vec<f64> {
        let rel = tss_row(row, &self.kept);
        match self.config.transform {
            Transform::None => self.kept.iter().map(|&j| row[j]).collect(),
            Transform::Tss => rel,
            Transform::LogTss => rel.iter().map(|v| (v + self.pseudocount).ln()).collect(),
            Transform::ArcsinSqrt => rel.iter().map(|v| v.sqrt().asin()).collect(),
            Transform::Clr => clr(&rel, self.pseudocount),
            Transform::PresenceAbsence => self
                .kept
                .iter()
                .map(|&j| if row[j] > 0.0 { 1.0 } else { 0.0 })
                .collect(),
            Transform::Ilr => {
                let logs: Vec<f64> = rel.iter().map(|v| (v + self.pseudocount).ln()).collect();
                self.balances.as_ref().expect("fitted").compute(&logs)
            }
            Transform::CladeClr => {
                let agg: Vec<f64> = self
                    .groups
                    .iter()
                    .map(|g| g.iter().map(|&k| rel[k]).sum())
                    .collect();
                clr(&agg, self.pseudocount)
            }
        }
    }

    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(MicrobiomeError::InvalidInput(
                "preprocessor not fitted".into(),
            ));
        }
        let p = x.first().map(|r| r.len()).unwrap_or(0);
        if self.kept.iter().any(|&j| j >= p) {
            return Err(MicrobiomeError::InvalidInput(
                "table has fewer columns than the training table".into(),
            ));
        }
        Ok(x.iter()
            .map(|r| {
                let mut v = self.apply(r);
                if self.config.standardize {
                    for (j, e) in v.iter_mut().enumerate() {
                        *e = (*e - self.means[j]) / self.stds[j];
                    }
                }
                v
            })
            .collect())
    }

    pub fn fit_transform(&mut self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        self.fit(x)?;
        self.transform(x)
    }

    /// Names of the output features. For `Ilr`/`CladeClr` these are balances /
    /// clades, not single taxa; `feature_names` are the original column names.
    pub fn output_names(&self, feature_names: &[String]) -> Vec<String> {
        let kept_names: Vec<String> = self
            .kept
            .iter()
            .map(|&j| feature_names[j].clone())
            .collect();
        match self.config.transform {
            Transform::Ilr => self
                .balances
                .as_ref()
                .map(|b| b.labels(&kept_names))
                .unwrap_or_default(),
            Transform::CladeClr => self
                .groups
                .iter()
                .enumerate()
                .map(|(g, m)| format!("clade{g}[{}+{}]", kept_names[m[0]], m.len() - 1))
                .collect(),
            _ => kept_names,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phylo::PhyloTree;

    #[test]
    fn clr_rows_sum_to_zero() {
        let c = clr(&[0.2, 0.3, 0.5], 1e-6);
        assert!(c.iter().sum::<f64>().abs() < 1e-12);
    }

    #[test]
    fn pipeline_filters_and_transforms() {
        let x = vec![
            vec![10.0, 0.0, 5.0, 85.0],
            vec![20.0, 0.0, 0.0, 80.0],
            vec![30.0, 1.0, 10.0, 59.0],
            vec![5.0, 0.0, 5.0, 90.0],
        ];
        let mut p = Preprocessor::new(PreprocessorConfig {
            min_prevalence: 0.5,
            ..Default::default()
        });
        let z = p.fit_transform(&x).unwrap();
        assert_eq!(p.kept, vec![0, 2, 3]);
        for r in z {
            assert!(r.iter().sum::<f64>().abs() < 1e-9);
        }
    }

    fn table() -> (Vec<Vec<f64>>, Vec<String>) {
        let names: Vec<String> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let x = vec![
            vec![10.0, 20.0, 5.0, 40.0, 25.0],
            vec![12.0, 18.0, 9.0, 30.0, 31.0],
            vec![8.0, 25.0, 4.0, 45.0, 18.0],
            vec![30.0, 5.0, 12.0, 10.0, 43.0],
            vec![25.0, 9.0, 15.0, 12.0, 39.0],
            vec![11.0, 22.0, 6.0, 35.0, 26.0],
        ];
        (x, names)
    }

    #[test]
    fn ilr_with_tree_has_p_minus_one_balances_and_preserves_clr_geometry() {
        let (x, names) = table();
        let tree = PhyloTree::from_newick("(((a:1,b:1):1,c:2):1,(d:1,e:1):2);").unwrap();
        let cfg = |t| PreprocessorConfig {
            transform: t,
            min_prevalence: 0.0,
            phylogeny: Some(Phylogeny::new(tree.clone(), &names)),
            ..Default::default()
        };
        let mut ilr = Preprocessor::new(cfg(Transform::Ilr));
        let zi = ilr.fit_transform(&x).unwrap();
        let mut c = Preprocessor::new(cfg(Transform::Clr));
        let zc = c.fit_transform(&x).unwrap();
        assert_eq!(zi[0].len(), 4);
        let d =
            |a: &Vec<f64>, b: &Vec<f64>| a.iter().zip(b).map(|(u, v)| (u - v).powi(2)).sum::<f64>();
        assert!((d(&zi[0], &zi[3]) - d(&zc[0], &zc[3])).abs() < 1e-9);
        assert_eq!(ilr.output_names(&names).len(), 4);
    }

    #[test]
    fn ilr_without_tree_and_clade_clr_work_and_missing_taxa_are_dropped() {
        let (x, names) = table();
        let mut ilr = Preprocessor::new(PreprocessorConfig {
            transform: Transform::Ilr,
            min_prevalence: 0.0,
            ..Default::default()
        });
        assert_eq!(ilr.fit_transform(&x).unwrap()[0].len(), 4);

        let tree = PhyloTree::from_newick("(((a:1,b:1):1,c:2):1,(d:1,x:1):2);").unwrap(); // 'e' missing, 'x' extra
        let ph = Phylogeny::new(tree, &names);
        let mut cc = Preprocessor::new(PreprocessorConfig {
            transform: Transform::CladeClr,
            min_prevalence: 0.0,
            clade_height_fraction: 0.4,
            phylogeny: Some(ph),
            ..Default::default()
        });
        let z = cc.fit_transform(&x).unwrap();
        assert_eq!(cc.unplaced, 1);
        assert!(z[0].len() >= 2 && z[0].len() < 4);
        assert!(z[0].iter().sum::<f64>().abs() < 1e-9);
    }
}
