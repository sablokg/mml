//! Evaluates the v0.2 features on simulated cohorts with a known ground truth.
//! cargo run --release --features xgboost --example feature_bench -- <out.csv> [reps]
use microbiome_ml::cv::{kfold, take, Lcg};
use microbiome_ml::metrics;
use microbiome_ml::models::FittedClassifier;
use microbiome_ml::prelude::*;
use std::fmt::Write as _;
use std::time::Instant;

struct Rng(Lcg);
impl Rng {
    fn u(&mut self) -> f64 { ((self.0.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64 }
    fn normal(&mut self) -> f64 { (-2.0 * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos() }
    fn poisson(&mut self, l: f64) -> f64 {
        if l > 30.0 { return (l + l.sqrt() * self.normal()).round().max(0.0); }
        let (lim, mut k, mut p) = ((-l).exp(), 0.0, 1.0);
        loop { p *= self.u(); if p <= lim { return k; } k += 1.0; }
    }
}

/// Balanced-by-halves tree over taxa lo..hi with random branch lengths.
fn newick(lo: usize, hi: usize, r: &mut Rng, root: bool) -> String {
    let len = 0.5 + r.u();
    let body = if hi - lo == 1 { format!("t{lo}") } else { let mid = (lo + hi) / 2; format!("({},{})", newick(lo, mid, r, false), newick(mid, hi, r, false)) };
    if root { format!("{body};") } else { format!("{body}:{len:.3}") }
}

/// signal: (taxon index, log-shift in group 1).
fn simulate(n: usize, p: usize, signal: &[(usize, f64)], seed: u64) -> (Vec<Vec<f64>>, Vec<u32>) {
    let mut r = Rng(Lcg::new(seed));
    let mu: Vec<f64> = (0..p).map(|_| 2.0 * r.normal()).collect();
    let (mut x, mut y) = (vec![], vec![]);
    for i in 0..n {
        let g = (i % 2) as u32;
        let depth = (10.0 + 0.5 * r.normal()).exp();
        let w: Vec<f64> = (0..p)
            .map(|j| {
                let shift = signal.iter().find(|s| s.0 == j).map(|s| s.1).unwrap_or(0.0) * g as f64;
                (mu[j] + shift + 0.5 * r.normal()).exp()
            })
            .collect();
        let tot: f64 = w.iter().sum();
        x.push(w.iter().map(|v| r.poisson(depth * v / tot)).collect());
        y.push(g);
    }
    (x, y)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out_path = args.get(1).cloned().unwrap_or("feature_bench.csv".into());
    let reps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let (n, p) = (120usize, 100usize);
    // coherent: one clade (taxa 0..6, all enriched); scattered: 6 taxa spread over the tree, mixed signs
    let scenarios: Vec<(&str, Vec<(usize, f64)>)> = vec![
        ("clade", (0..6).map(|j| (j, 1.0)).collect()),
        ("scattered", vec![(0, 1.0), (17, -1.0), (34, 1.0), (51, -1.0), (68, 1.0), (85, -1.0)]),
    ];
    let mut csv = String::from("scenario,rep,method,value,secs\n");
    for (sname, signal) in &scenarios {
        let truth: Vec<usize> = signal.iter().map(|s| s.0).collect();
        let k = truth.len();
        for rep in 0..reps {
            let seed = 9000 + 100 * rep as u64 + sname.len() as u64;
            let (x, y) = simulate(n, p, signal, seed);
            let names: Vec<String> = (0..p).map(|j| format!("t{j}")).collect();
            let tree = PhyloTree::from_newick(&newick(0, p, &mut Rng(Lcg::new(seed + 1)), true)).unwrap();
            let ph = Phylogeny::new(tree.clone(), &names);
            let (tr, te) = kfold(n, 3, Some(&y), seed).remove(0);
            let (xtr, xte, ytr, yte) = (take(&x, &tr), take(&x, &te), take(&y, &tr), take(&y, &te));
            let labels = |v: &[u32]| -> Vec<String> { v.iter().map(|c| format!("c{c}")).collect() };
            let ds = Dataset::new(xtr.clone(), Some(names.clone()), None).unwrap();
            let mut log = |m: &str, v: f64, s: f64| { let _ = writeln!(csv, "{sname},{rep},{m},{v},{s}"); };

            // --- fixed pipelines on the held-out test set
            let fixed = |t: Transform, kind: ClassifierKind, tree: bool| -> Option<(f64, f64)> {
                let t0 = Instant::now();
                let cfg = PreprocessorConfig { transform: t, standardize: kind.wants_scaling(), phylogeny: if tree { Some(ph.clone()) } else { None }, ..Default::default() };
                let mut pre = Preprocessor::new(cfg);
                let ztr = pre.fit_transform(&xtr).ok()?;
                let zte = pre.transform(&xte).ok()?;
                let pred = FittedClassifier::fit(kind, &ztr, &ytr, 2, 42).ok()?.predict(&zte).ok()?;
                Some((metrics::balanced_accuracy(&yte, &pred, 2), t0.elapsed().as_secs_f64()))
            };
            for (name, t, kind, tree) in [
                ("CLR+LR", Transform::Clr, ClassifierKind::LogisticRegression, false),
                ("ILRdata+LR", Transform::Ilr, ClassifierKind::LogisticRegression, false),
                ("ILRphylo+LR", Transform::Ilr, ClassifierKind::LogisticRegression, true),
                ("CladeCLR+LR", Transform::CladeClr, ClassifierKind::LogisticRegression, true),
                ("CLR+RF", Transform::Clr, ClassifierKind::RandomForest, false),
                ("ILRdata+RF", Transform::Ilr, ClassifierKind::RandomForest, false),
                ("ILRphylo+RF", Transform::Ilr, ClassifierKind::RandomForest, true),
                ("CLR+XGB", Transform::Clr, ClassifierKind::XgBoost, false),
                ("ILRphylo+XGB", Transform::Ilr, ClassifierKind::XgBoost, true),
            ] {
                if let Some((v, s)) = fixed(t, kind, tree) { log(name, v, s); } else { log(name, f64::NAN, 0.0); }
            }

            // --- AutoML variants (selection on training data only; scored on test)
            let base = || { let mut a = AutoClassifier::default(); a.transforms = vec![Transform::Clr, Transform::LogTss, Transform::Tss]; a.models = vec![ClassifierKind::LogisticRegression, ClassifierKind::RandomForest, ClassifierKind::XgBoost]; a };
            for (name, auto) in [
                ("AUTO_base", base()),
                ("AUTO_grid", base().with_grid(GridLevel::Small)),
                ("AUTO_phylo", base().with_phylogeny(tree.clone(), &names)),
            ] {
                let t0 = Instant::now();
                let fit = auto.fit(&ds, &labels(&ytr)).unwrap();
                let secs = t0.elapsed().as_secs_f64();
                let pred = fit.predict_indices(&xte).unwrap();
                log(name, metrics::balanced_accuracy(&yte, &pred, 2), secs);
                log(&format!("{name}:choice={:?}+{:?}[{}]", fit.best.transform, fit.best.model, fit.best.params).replace(',', ";"), 0.0, 0.0);
            }

            // --- importance recovery: top-k precision against the true signal taxa
            let prec = |ranked: Vec<usize>| ranked.iter().take(k).filter(|c| truth.contains(c)).count() as f64 / k as f64;
            let col = |name: &str| name[1..].parse::<usize>().unwrap();
            for (name, kind) in [("RF", ClassifierKind::RandomForest), ("XGB", ClassifierKind::XgBoost)] {
                let mut a = AutoClassifier::default();
                a.transforms = vec![Transform::Clr];
                a.models = vec![kind];
                let fit = a.fit(&ds, &labels(&ytr)).unwrap();
                let imp = fit.taxon_importance(&xte, &labels(&yte), 5, seed).unwrap();
                log(&format!("importance_perm_{name}_p@{k}"), prec(imp.iter().map(|t| t.column).collect()), 0.0);
                if let Some(nat) = fit.native_importance() { log(&format!("importance_native_{name}_p@{k}"), prec(nat.iter().map(|(n, _)| col(n)).collect()), 0.0); }
            }
            eprintln!("done {sname} rep {rep}");
        }
        std::fs::write(&out_path, &csv).unwrap();
    }
    std::fs::write(&out_path, &csv).unwrap();
}
