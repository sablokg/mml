//! Real-data benchmark (Zhu et al. 2020, https://www.nature.com/articles/s41467-020-15457-9): repeated stratified 5-fold CV on a labelled count table.
//! cargo run --release --features xgboost --example zhu_bench -- <table.tsv> <out.csv> <folds.csv> [reps] [auto:0|1]
//! Table: first column sample id, last column `__label__`, other columns taxa (raw counts).
use microbiome_ml::cv::{kfold, take};
use microbiome_ml::metrics;
use microbiome_ml::models::FittedClassifier;
use microbiome_ml::prelude::*;
use std::fmt::Write as _;
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (path, out, folds_out) = (&a[1], &a[2], &a[3]);
    let reps: usize = a.get(4).and_then(|s| s.parse().ok()).unwrap_or(3);
    let run_auto = a.get(5).map(|s| s != "0").unwrap_or(true);

    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split('\t').collect();
    let names: Vec<String> = header[1..header.len() - 1].iter().map(|s| s.to_string()).collect();
    let (mut x, mut lab) = (vec![], vec![]);
    for l in lines {
        let f: Vec<&str> = l.split('\t').collect();
        x.push(f[1..f.len() - 1].iter().map(|v| v.parse::<f64>().unwrap()).collect::<Vec<f64>>());
        lab.push(f[f.len() - 1].to_string());
    }
    let n = x.len();
    let classes = { let mut c = lab.clone(); c.sort(); c.dedup(); c };
    let y: Vec<u32> = lab.iter().map(|l| classes.iter().position(|c| c == l).unwrap() as u32).collect();
    eprintln!("n={n} p={} classes={classes:?}", names.len());

    let mut csv = String::from("rep,fold,method,bal_acc,secs\n");
    let mut fcsv = String::from("rep,fold,test_idx\n");
    for rep in 0..reps {
        let seed = 1000 + rep as u64;
        for (fi, (tr, te)) in kfold(n, 5, Some(&y), seed).into_iter().enumerate() {
            let idx: Vec<String> = te.iter().map(|i| i.to_string()).collect();
            let _ = writeln!(fcsv, "{rep},{fi},{}", idx.join(" "));
            let (xtr, xte, ytr, yte) = (take(&x, &tr), take(&x, &te), take(&y, &tr), take(&y, &te));
            let ltr: Vec<String> = ytr.iter().map(|&c| classes[c as usize].clone()).collect();
            let mut log = |m: &str, v: f64, s: f64| { let _ = writeln!(csv, "{rep},{fi},{m},{v},{s}"); };

            for (name, t, kind) in [
                ("CLR+LR", Transform::Clr, ClassifierKind::LogisticRegression),
                ("CLR+RF", Transform::Clr, ClassifierKind::RandomForest),
                ("CLR+XGB", Transform::Clr, ClassifierKind::XgBoost),
                ("LogTss+RF", Transform::LogTss, ClassifierKind::RandomForest),
                ("Tss+RF", Transform::Tss, ClassifierKind::RandomForest),
                ("PA+RF", Transform::PresenceAbsence, ClassifierKind::RandomForest),
                ("ILRdata+LR", Transform::Ilr, ClassifierKind::LogisticRegression),
            ] {
                let t0 = Instant::now();
                let cfg = PreprocessorConfig { transform: t, standardize: kind.wants_scaling(), ..Default::default() };
                let mut pre = Preprocessor::new(cfg);
                let r = (|| -> Result<f64> {
                    let ztr = pre.fit_transform(&xtr)?;
                    let zte = pre.transform(&xte)?;
                    let pred = FittedClassifier::fit(kind, &ztr, &ytr, 2, 42)?.predict(&zte)?;
                    Ok(metrics::balanced_accuracy(&yte, &pred, 2))
                })();
                match r { Ok(v) => log(name, v, t0.elapsed().as_secs_f64()), Err(e) => { eprintln!("{name}: {e}"); log(name, f64::NAN, 0.0) } }
            }
            if run_auto {
                let ds = Dataset::new(xtr.clone(), Some(names.clone()), None).unwrap();
                let t0 = Instant::now();
                let fit = AutoClassifier::default().fit(&ds, &ltr).unwrap();
                let secs = t0.elapsed().as_secs_f64();
                let pred = fit.predict_indices(&xte).unwrap();
                log("AUTO", metrics::balanced_accuracy(&yte, &pred, 2), secs);
                log(&format!("AUTO:choice={:?}+{:?}", fit.best.transform, fit.best.model), 0.0, 0.0);
            }
            eprintln!("rep {rep} fold {fi} done");
            std::fs::write(out, &csv).unwrap();
        }
    }
    std::fs::write(out, &csv).unwrap();
    std::fs::write(folds_out, &fcsv).unwrap();
}
