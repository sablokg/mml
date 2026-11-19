//! Benchmark harness. Usage: cargo run --release --example benchmark -- <outdir> [quick]
//! Writes results.csv, scaling.csv and per-dataset exports for the Python cross-check.
use microbiome_ml::cv::{kfold, take, Lcg};
use microbiome_ml::metrics;
use microbiome_ml::models::{ClassifierKind, FittedClassifier, FittedRegressor, RegressorKind};
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

/// Simulated cohort. Binary mode: group shift in `nd` taxa (half up, half down).
/// Continuous mode: latent gradient z loads on `nd` taxa; target = z + noise.
fn simulate(n: usize, p: usize, effect: f64, continuous: bool, seed: u64) -> (Vec<Vec<f64>>, Vec<u32>, Vec<f64>) {
    let mut r = Rng(Lcg::new(seed));
    let nd = (p / 20).max(5);
    let mu: Vec<f64> = (0..p).map(|_| 2.0 * r.normal()).collect();
    let load: Vec<f64> = (0..p).map(|j| if j < nd { if j % 2 == 0 { effect } else { -effect } } else { 0.0 }).collect();
    let (mut x, mut yc, mut yr) = (vec![], vec![], vec![]);
    for i in 0..n {
        let g = (i % 2) as u32;
        let z = r.normal();
        let depth = (10.0 + 0.5 * r.normal()).exp(); // median ~22k reads
        let w: Vec<f64> = (0..p).map(|j| {
            let shift = if continuous { load[j] * z * 0.5 } else { load[j] * g as f64 };
            (mu[j] + shift + 0.5 * r.normal()).exp()
        }).collect();
        let tot: f64 = w.iter().sum();
        x.push(w.iter().map(|v| r.poisson(depth * v / tot)).collect());
        yc.push(g);
        yr.push(z + 0.5 * r.normal());
    }
    (x, yc, yr)
}

fn csv_export(dir: &str, name: &str, x: &[Vec<f64>], y: &[f64], test: &[usize]) {
    let mut s = String::new();
    for (i, row) in x.iter().enumerate() {
        let t = test.contains(&i) as u8;
        let _ = writeln!(s, "{},{},{}", t, y[i], row.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","));
    }
    std::fs::write(format!("{dir}/{name}.csv"), s).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = args.get(1).cloned().unwrap_or("bench_out".into());
    let quick = args.get(2).map(|s| s == "quick").unwrap_or(false);
    std::fs::create_dir_all(format!("{dir}/data")).unwrap();
    let reps = if quick { 2 } else { 8 };
    let scen: Vec<(usize, usize, f64)> = if quick {
        vec![(100, 100, 1.0)]
    } else {
        vec![(60, 100, 1.0), (120, 100, 1.0), (120, 100, 0.5), (240, 300, 0.5), (120, 500, 0.5)]
    };
    let offset: u64 = std::env::var("BENCH_SEED_OFFSET").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let only: Option<usize> = std::env::var("BENCH_SCEN").ok().and_then(|v| v.parse().ok());
    let noscale = std::env::var("BENCH_NOSCALE").is_ok();
    let mut out = String::from("task,n,p,effect,rep,method,score,secondary,fit_s,kept\n");

    for (si, &(n, p, eff)) in scen.iter().enumerate() {
        if only.map_or(false, |o| o != si) { continue; }
        for rep in 0..reps {
            let seed = 1000 * rep as u64 + n as u64 + p as u64 + offset;
            for continuous in [false, true] {
                let (x, yc, yr) = simulate(n, p, eff, continuous, seed);
                let task = if continuous { "reg" } else { "clf" };
                let strata = if continuous { None } else { Some(&yc[..]) };
                let (tr, te) = kfold(n, 3, strata, seed).remove(0);
                let (xtr, xte) = (take(&x, &tr), take(&x, &te));
                let yv: Vec<f64> = if continuous { yr.clone() } else { yc.iter().map(|&v| v as f64).collect() };
                csv_export(&format!("{dir}/data"), &format!("{task}_n{n}_p{p}_e{eff}_r{rep}"), &x, &yv, &te);
                let base = PreprocessorConfig::default();
                let t0 = Instant::now();
                if !continuous {
                    let (ytr, yte) = (take(&yc, &tr), take(&yc, &te));
                    for &t in &[Transform::None, Transform::Tss, Transform::Clr, Transform::LogTss, Transform::ArcsinSqrt, Transform::PresenceAbsence] {
                        for m in ClassifierKind::all() {
                            let t1 = Instant::now();
                            let mut pre = Preprocessor::new(PreprocessorConfig { transform: t, standardize: m.wants_scaling(), ..base.clone() });
                            let (Ok(ztr), ) = (pre.fit_transform(&xtr), ) else { let _ = writeln!(out, "{task},{n},{p},{eff},{rep},{t:?}+{m:?},NaN,NaN,0,0"); continue };
                            let zte = pre.transform(&xte).unwrap();
                            let Ok(fm) = FittedClassifier::fit(m, &ztr, &ytr, 2, 42) else { let _ = writeln!(out, "{task},{n},{p},{eff},{rep},{t:?}+{m:?},NaN,NaN,0,0"); continue };
                            let pred = fm.predict(&zte).unwrap();
                            let _ = writeln!(out, "{task},{n},{p},{eff},{rep},{t:?}+{m:?},{},{},{},{}", metrics::balanced_accuracy(&yte, &pred, 2), metrics::macro_f1(&yte, &pred, 2), t1.elapsed().as_secs_f64(), pre.kept.len());
                        }
                    }
                    let labels: Vec<String> = ytr.iter().map(|v| format!("c{v}")).collect();
                    let ds = Dataset::new(xtr.clone(), None, None).unwrap();
                    let t1 = Instant::now();
                    let fit = AutoClassifier::default().fit(&ds, &labels).unwrap();
                    let el = t1.elapsed().as_secs_f64();
                    let pred = fit.predict_indices(&xte).unwrap();
                    let _ = writeln!(out, "{task},{n},{p},{eff},{rep},AUTO[{:?}+{:?}],{},{},{},{}", fit.best.transform, fit.best.model, metrics::balanced_accuracy(&yte, &pred, 2), metrics::macro_f1(&yte, &pred, 2), el, fit.preprocessor.kept.len());
                } else {
                    let (ytr, yte) = (take(&yr, &tr), take(&yr, &te));
                    for &t in &[Transform::None, Transform::Tss, Transform::Clr, Transform::LogTss, Transform::ArcsinSqrt, Transform::PresenceAbsence] {
                        for m in RegressorKind::all() {
                            let t1 = Instant::now();
                            let mut pre = Preprocessor::new(PreprocessorConfig { transform: t, standardize: m.wants_scaling(), ..base.clone() });
                            let (Ok(ztr), ) = (pre.fit_transform(&xtr), ) else { let _ = writeln!(out, "{task},{n},{p},{eff},{rep},{t:?}+{m:?},NaN,NaN,0,0"); continue };
                            let zte = pre.transform(&xte).unwrap();
                            let Ok(fm) = FittedRegressor::fit(m, &ztr, &ytr, 42) else { let _ = writeln!(out, "{task},{n},{p},{eff},{rep},{t:?}+{m:?},NaN,NaN,0,0"); continue };
                            let pred = fm.predict(&zte).unwrap();
                            let _ = writeln!(out, "{task},{n},{p},{eff},{rep},{t:?}+{m:?},{},{},{},{}", metrics::r2(&yte, &pred), metrics::rmse(&yte, &pred), t1.elapsed().as_secs_f64(), pre.kept.len());
                        }
                    }
                    let ds = Dataset::new(xtr.clone(), None, None).unwrap();
                    let t1 = Instant::now();
                    let fit = AutoRegressor::default().fit(&ds, &ytr).unwrap();
                    let el = t1.elapsed().as_secs_f64();
                    let pred = fit.predict(&xte).unwrap();
                    let _ = writeln!(out, "{task},{n},{p},{eff},{rep},AUTO[{:?}+{:?}],{},{},{},{}", fit.best.transform, fit.best.model, metrics::r2(&yte, &pred), metrics::rmse(&yte, &pred), el, fit.preprocessor.kept.len());
                }
                eprintln!("done {task} n={n} p={p} e={eff} rep={rep} ({:.1}s)", t0.elapsed().as_secs_f64());
            }
        }
        std::fs::write(format!("{dir}/results.csv"), &out).unwrap();
    }
    if noscale { return; }

    // Scaling: wall-clock of preprocessing and single fits.
    let mut sc = String::from("n,p,stage,secs\n");
    let sizes: Vec<(usize, usize)> = if quick { vec![(100, 100)] } else { vec![(100, 500), (200, 500), (400, 500), (800, 500), (400, 100), (400, 1000), (400, 2000)] };
    for (n, p) in sizes {
        let (x, yc, yr) = simulate(n, p, 0.5, false, 5);
        let t = Instant::now();
        let mut pre = Preprocessor::new(PreprocessorConfig::default());
        let z = pre.fit_transform(&x).unwrap();
        let _ = writeln!(sc, "{n},{p},preprocess_clr,{}", t.elapsed().as_secs_f64());
        for m in ClassifierKind::all() {
            let t = Instant::now();
            let _ = FittedClassifier::fit(m, &z, &yc, 2, 1);
            let _ = writeln!(sc, "{n},{p},clf_{m:?},{}", t.elapsed().as_secs_f64());
        }
        for m in RegressorKind::all() {
            let t = Instant::now();
            let _ = FittedRegressor::fit(m, &z, &yr, 1);
            let _ = writeln!(sc, "{n},{p},reg_{m:?},{}", t.elapsed().as_secs_f64());
        }
        eprintln!("scaling n={n} p={p} done");
    }
    std::fs::write(format!("{dir}/scaling.csv"), sc).unwrap();
}
