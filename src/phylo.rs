//! Phylogeny support: Newick parsing, sequential binary partitions (ILR
//! balances), clade grouping, and a data-driven dendrogram fallback.
//!
//! Balances follow Egozcue et al. (2003): for a node with numerator taxa R
//! (size r) and denominator taxa S (size s),
//! `b = sqrt(rs/(r+s)) * (mean(ln x_R) - mean(ln x_S))`.
//! The full set of p-1 balances is an isometry of CLR space, so Euclidean
//! distances between balance vectors equal Aitchison distances.
//!
//! Polytomies are resolved arbitrarily (children split in halves, in order).

use crate::error::{MicrobiomeError, Result};
use std::collections::HashMap;
use std::sync::Arc;

/*
Gaurav Sablok
gsablok@proton.me
*/

/// Largest number of taxa for the data-driven dendrogram (cubic-time clustering).
pub const MAX_DENDROGRAM_TAXA: usize = 2000;

#[derive(Debug, Clone)]
pub struct Clade {
    pub name: Option<String>,
    /// Branch length to the parent (0 if absent).
    pub length: f64,
    pub children: Vec<Clade>,
}

impl Clade {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
    /// Longest path from this node down to a leaf.
    pub fn height(&self) -> f64 {
        self.children
            .iter()
            .map(|c| c.length.max(0.0) + c.height())
            .fold(0.0, f64::max)
    }
    fn leaf_names<'a>(&'a self, out: &mut Vec<&'a str>) {
        if self.is_leaf() {
            if let Some(n) = &self.name {
                out.push(n);
            }
        } else {
            for c in &self.children {
                c.leaf_names(out);
            }
        }
    }
    fn any_length(&self) -> bool {
        self.length > 0.0 || self.children.iter().any(|c| c.any_length())
    }
}

#[derive(Debug, Clone)]
pub struct PhyloTree {
    pub root: Clade,
}

impl PhyloTree {
    pub fn from_newick(s: &str) -> Result<Self> {
        let mut p = Parser {
            s: s.as_bytes(),
            i: 0,
        };
        let root = p.subtree()?;
        p.ws();
        if p.peek() == Some(b';') {
            p.i += 1;
        }
        p.ws();
        if p.i != p.s.len() {
            return Err(p.err("trailing characters after tree"));
        }
        let t = Self { root };
        let mut names = Vec::new();
        t.root.leaf_names(&mut names);
        if names.len() < 2 {
            return Err(MicrobiomeError::InvalidInput(
                "tree needs at least two named leaves".into(),
            ));
        }
        Ok(t)
    }

    pub fn from_newick_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        Self::from_newick(&std::fs::read_to_string(path)?)
    }

    pub fn leaf_names(&self) -> Vec<String> {
        let mut v = Vec::new();
        self.root.leaf_names(&mut v);
        v.into_iter().map(String::from).collect()
    }

    pub fn has_branch_lengths(&self) -> bool {
        self.root.any_length()
    }

    pub fn height(&self) -> f64 {
        self.root.height()
    }
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, m: &str) -> MicrobiomeError {
        MicrobiomeError::InvalidInput(format!("newick: {m} at byte {}", self.i))
    }
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }
    fn ws(&mut self) {
        while self.peek().map_or(false, |c| (c as char).is_whitespace()) {
            self.i += 1;
        }
    }
    fn subtree(&mut self) -> Result<Clade> {
        self.ws();
        let mut children = Vec::new();
        if self.peek() == Some(b'(') {
            self.i += 1;
            loop {
                children.push(self.subtree()?);
                self.ws();
                match self.peek() {
                    Some(b',') => self.i += 1,
                    Some(b')') => {
                        self.i += 1;
                        break;
                    }
                    _ => return Err(self.err("expected ',' or ')'")),
                }
            }
        }
        self.ws();
        let name = self.label()?;
        self.ws();
        let mut length = 0.0;
        if self.peek() == Some(b':') {
            self.i += 1;
            self.ws();
            let st = self.i;
            while self
                .peek()
                .map_or(false, |c| b"0123456789.eE+-".contains(&c))
            {
                self.i += 1;
            }
            length = std::str::from_utf8(&self.s[st..self.i])
                .unwrap_or("")
                .parse::<f64>()
                .map_err(|_| self.err("bad branch length"))?;
        }
        if children.is_empty() && name.is_empty() {
            return Err(self.err("unnamed leaf"));
        }
        Ok(Clade {
            name: if name.is_empty() { None } else { Some(name) },
            length,
            children,
        })
    }
    fn label(&mut self) -> Result<String> {
        if self.peek() == Some(b'\'') {
            self.i += 1;
            let mut out = Vec::new();
            loop {
                match self.peek() {
                    None => return Err(self.err("unterminated quoted label")),
                    Some(b'\'') => {
                        if self.s.get(self.i + 1) == Some(&b'\'') {
                            out.push(b'\'');
                            self.i += 2;
                        } else {
                            self.i += 1;
                            break;
                        }
                    }
                    Some(c) => {
                        out.push(c);
                        self.i += 1;
                    }
                }
            }
            return Ok(String::from_utf8_lossy(&out).into_owned());
        }
        let st = self.i;
        while self.peek().map_or(false, |c| {
            !b",():;".contains(&c) && !(c as char).is_whitespace()
        }) {
            self.i += 1;
        }
        Ok(String::from_utf8_lossy(&self.s[st..self.i]).into_owned())
    }
}

/// A tree tied to the dataset's feature (column) names.
#[derive(Debug, Clone)]
pub struct Phylogeny {
    pub tree: Arc<PhyloTree>,
    /// Feature name of every column of the training table.
    pub names: Arc<Vec<String>>,
}

impl Phylogeny {
    pub fn new(tree: PhyloTree, feature_names: &[String]) -> Self {
        Self {
            tree: Arc::new(tree),
            names: Arc::new(feature_names.to_vec()),
        }
    }
    /// Number of dataset columns whose name is a leaf of the tree.
    pub fn n_matched(&self) -> usize {
        let leaves: std::collections::HashSet<String> =
            self.tree.leaf_names().into_iter().collect();
        self.names.iter().filter(|n| leaves.contains(*n)).count()
    }
}

/// Binary partition over positions `0..k`.
#[derive(Debug, Clone)]
pub enum Bin {
    Leaf(usize),
    Join(Box<Bin>, Box<Bin>),
}

fn join_all(mut v: Vec<Bin>) -> Option<Bin> {
    match v.len() {
        0 => None,
        1 => v.pop(),
        n => {
            let right = v.split_off(n / 2);
            Some(Bin::Join(
                Box::new(join_all(v)?),
                Box::new(join_all(right)?),
            ))
        }
    }
}

/// Restrict a clade to leaves present in `pos` (name -> position) and binarise.
pub fn bin_from_clade(c: &Clade, pos: &HashMap<&str, usize>) -> Option<Bin> {
    if c.is_leaf() {
        return c
            .name
            .as_deref()
            .and_then(|n| pos.get(n))
            .map(|&i| Bin::Leaf(i));
    }
    join_all(
        c.children
            .iter()
            .filter_map(|k| bin_from_clade(k, pos))
            .collect(),
    )
}

fn leaves_in(c: &Clade, pos: &HashMap<&str, usize>, out: &mut Vec<usize>) {
    if c.is_leaf() {
        if let Some(&i) = c.name.as_deref().and_then(|n| pos.get(n)) {
            out.push(i);
        }
    } else {
        for k in &c.children {
            leaves_in(k, pos, out);
        }
    }
}

/// Maximal clades whose height (longest path to a tip) is at most `cut`.
pub fn clade_groups(root: &Clade, pos: &HashMap<&str, usize>, cut: f64) -> Vec<Vec<usize>> {
    fn rec(c: &Clade, pos: &HashMap<&str, usize>, cut: f64, out: &mut Vec<Vec<usize>>) {
        if c.height() <= cut {
            let mut g = Vec::new();
            leaves_in(c, pos, &mut g);
            if !g.is_empty() {
                out.push(g);
            }
        } else {
            for k in &c.children {
                rec(k, pos, cut, out);
            }
        }
    }
    let mut out = Vec::new();
    rec(root, pos, cut, &mut out);
    out
}

#[derive(Debug, Clone, Copy)]
enum Child {
    Leaf(usize),
    Node(usize),
}

#[derive(Debug, Clone)]
struct Rec {
    l: Child,
    r: Child,
    nl: usize,
    nr: usize,
    first_l: usize,
    first_r: usize,
}

/// Flattened sequential binary partition, children before parents.
#[derive(Debug, Clone)]
pub struct Balances {
    nodes: Vec<Rec>,
}

impl Balances {
    pub fn from_bin(b: &Bin) -> Self {
        fn flat(b: &Bin, nodes: &mut Vec<Rec>) -> (Child, usize, usize) {
            match b {
                Bin::Leaf(i) => (Child::Leaf(*i), 1, *i),
                Bin::Join(l, r) => {
                    let (cl, nl, fl) = flat(l, nodes);
                    let (cr, nr, fr) = flat(r, nodes);
                    nodes.push(Rec {
                        l: cl,
                        r: cr,
                        nl,
                        nr,
                        first_l: fl,
                        first_r: fr,
                    });
                    (Child::Node(nodes.len() - 1), nl + nr, fl)
                }
            }
        }
        let mut nodes = Vec::new();
        flat(b, &mut nodes);
        Self { nodes }
    }

    pub fn n_balances(&self) -> usize {
        self.nodes.len()
    }

    /// `logs[i]` = ln(abundance) of taxon at position `i`.
    pub fn compute(&self, logs: &[f64]) -> Vec<f64> {
        let mut sums = vec![0.0; self.nodes.len()];
        let mut out = vec![0.0; self.nodes.len()];
        let val = |c: Child, sums: &[f64]| match c {
            Child::Leaf(i) => logs[i],
            Child::Node(k) => sums[k],
        };
        for (k, n) in self.nodes.iter().enumerate() {
            let (sl, sr) = (val(n.l, &sums), val(n.r, &sums));
            sums[k] = sl + sr;
            let (r, s) = (n.nl as f64, n.nr as f64);
            out[k] = (r * s / (r + s)).sqrt() * (sl / r - sr / s);
        }
        out
    }

    /// Human-readable labels; `names[i]` is the taxon at position `i`.
    pub fn labels(&self, names: &[String]) -> Vec<String> {
        self.nodes
            .iter()
            .enumerate()
            .map(|(k, n)| {
                format!(
                    "b{k}[{}+{} | {}+{}]",
                    names[n.first_l],
                    n.nl - 1,
                    names[n.first_r],
                    n.nr - 1
                )
            })
            .collect()
    }
}

/// Average-linkage dendrogram on the variation matrix tau_ij = var(ln x_i - ln x_j),
/// the standard data-driven balance dendrogram of compositional data analysis.
pub fn dendrogram_from_logs(logs: &[Vec<f64>]) -> Result<Bin> {
    let n = logs.len();
    let k = logs.first().map(|r| r.len()).unwrap_or(0);
    if n < 3 || k < 2 {
        return Err(MicrobiomeError::InvalidInput(
            "dendrogram needs >= 3 samples and >= 2 taxa".into(),
        ));
    }
    if k > MAX_DENDROGRAM_TAXA {
        return Err(MicrobiomeError::InvalidInput(format!("{k} taxa exceeds the data-driven ILR limit of {MAX_DENDROGRAM_TAXA}; raise min_prevalence or supply a phylogeny")));
    }
    let means: Vec<f64> = (0..k)
        .map(|j| logs.iter().map(|r| r[j]).sum::<f64>() / n as f64)
        .collect();
    let cen: Vec<Vec<f64>> = (0..k)
        .map(|j| logs.iter().map(|r| r[j] - means[j]).collect())
        .collect();
    let mut cov = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..=i {
            let c: f64 = cen[i].iter().zip(&cen[j]).map(|(a, b)| a * b).sum::<f64>() / n as f64;
            cov[i * k + j] = c;
            cov[j * k + i] = c;
        }
    }
    let mut d = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            if i != j {
                d[i * k + j] = cov[i * k + i] + cov[j * k + j] - 2.0 * cov[i * k + j];
            }
        }
    }
    let mut nodes: Vec<Option<Bin>> = (0..k).map(|i| Some(Bin::Leaf(i))).collect();
    let mut size = vec![1usize; k];
    let mut active = vec![true; k];
    for _ in 0..k - 1 {
        let (mut bi, mut bj, mut bd) = (0, 0, f64::INFINITY);
        for i in 0..k {
            if !active[i] {
                continue;
            }
            for j in i + 1..k {
                if active[j] && d[i * k + j] < bd {
                    bd = d[i * k + j];
                    bi = i;
                    bj = j;
                }
            }
        }
        let (si, sj) = (size[bi] as f64, size[bj] as f64);
        for m in 0..k {
            if active[m] && m != bi && m != bj {
                let nd = (si * d[bi * k + m] + sj * d[bj * k + m]) / (si + sj);
                d[bi * k + m] = nd;
                d[m * k + bi] = nd;
            }
        }
        let a = nodes[bi].take().unwrap();
        let b = nodes[bj].take().unwrap();
        nodes[bi] = Some(Bin::Join(Box::new(a), Box::new(b)));
        size[bi] += size[bj];
        active[bj] = false;
    }
    nodes
        .into_iter()
        .flatten()
        .next()
        .ok_or_else(|| MicrobiomeError::Model("dendrogram construction failed".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_newick_with_lengths_and_quotes() {
        let t = PhyloTree::from_newick("((A:0.1,B:0.2)90:0.3,('C d':0.4,D:0.5):0.1);").unwrap();
        assert_eq!(t.leaf_names(), vec!["A", "B", "C d", "D"]);
        assert!(t.has_branch_lengths());
        assert!((t.height() - 0.6).abs() < 1e-12);
        assert!(PhyloTree::from_newick("(A,B").is_err());
    }

    fn clr_dist2(x: &[f64], y: &[f64]) -> f64 {
        let c = |v: &[f64]| {
            let l: Vec<f64> = v.iter().map(|a| a.ln()).collect();
            let m = l.iter().sum::<f64>() / l.len() as f64;
            l.into_iter().map(|a| a - m).collect::<Vec<_>>()
        };
        c(x).iter().zip(c(y)).map(|(a, b)| (a - b).powi(2)).sum()
    }

    #[test]
    fn balances_are_an_isometry_of_clr_space() {
        let t = PhyloTree::from_newick("(((a,b),c),(d,(e,f,g)));").unwrap();
        let names = t.leaf_names();
        let pos: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let bal = Balances::from_bin(&bin_from_clade(&t.root, &pos).unwrap());
        assert_eq!(bal.n_balances(), 6);
        let x = [1.0, 4.0, 2.0, 9.0, 0.5, 3.0, 7.0];
        let y = [3.0, 1.0, 8.0, 2.0, 5.0, 0.2, 1.5];
        let bx = bal.compute(&x.iter().map(|v: &f64| v.ln()).collect::<Vec<_>>());
        let by = bal.compute(&y.iter().map(|v: &f64| v.ln()).collect::<Vec<_>>());
        let d: f64 = bx.iter().zip(&by).map(|(a, b)| (a - b).powi(2)).sum();
        assert!((d - clr_dist2(&x, &y)).abs() < 1e-9);
    }

    #[test]
    fn dendrogram_balances_are_also_an_isometry_and_group_covarying_taxa() {
        // taxa 0,1 covary strongly; 2,3 covary strongly; the two pairs are independent.
        let mut s = 3u64;
        let mut r = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        let logs: Vec<Vec<f64>> = (0..60)
            .map(|_| {
                let (a, b) = (r() * 3.0, r() * 3.0);
                vec![a, a + 0.05 * r(), b, b + 0.05 * r()]
            })
            .collect();
        let bin = dendrogram_from_logs(&logs).unwrap();
        let bal = Balances::from_bin(&bin);
        assert_eq!(bal.n_balances(), 3);
        // leaf pairs {0,1} and {2,3} must each be joined before being joined to the other pair
        let names: Vec<String> = (0..4).map(|i| format!("t{i}")).collect();
        let labels = bal.labels(&names).join(" ");
        assert!(
            labels.contains("t0+0 | t1+0") || labels.contains("t1+0 | t0+0"),
            "{labels}"
        );
        assert!(
            labels.contains("t2+0 | t3+0") || labels.contains("t3+0 | t2+0"),
            "{labels}"
        );
        let (x, y) = (&logs[0], &logs[1]);
        let d: f64 = bal
            .compute(x)
            .iter()
            .zip(bal.compute(y))
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        let m = |v: &Vec<f64>| v.iter().sum::<f64>() / 4.0;
        let c: f64 = x
            .iter()
            .zip(y)
            .map(|(a, b)| ((a - m(x)) - (b - m(y))).powi(2))
            .sum();
        assert!((d - c).abs() < 1e-9);
    }

    #[test]
    fn clade_groups_cut_by_height() {
        let t = PhyloTree::from_newick("((A:1,B:1):1,(C:1,D:1):1);").unwrap();
        let names = t.leaf_names();
        let pos: HashMap<&str, usize> = names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        assert_eq!(clade_groups(&t.root, &pos, 1.0).len(), 2); // {A,B} and {C,D}
        assert_eq!(clade_groups(&t.root, &pos, 0.0).len(), 4);
    }
}
