use microbiome_ml::{metrics, prelude::*};
use smartcore::ensemble::random_forest_classifier::{RandomForestClassifier, RandomForestClassifierParameters};
use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::BTreeMap;
fn main() {
    let dir = std::env::args().nth(1).unwrap();
    let mut acc: BTreeMap<(String, String), Vec<f64>> = BTreeMap::new();
    let mut files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    for f in files {
        let name = f.file_name().unwrap().to_str().unwrap().to_string();
        let is_clf = name.starts_with("clf");
        let scen = name.rsplit_once("_r").unwrap().0.to_string();
        let txt = std::fs::read_to_string(&f).unwrap();
        let (mut xtr, mut ytr, mut xte, mut yte) = (vec![], vec![], vec![], vec![]);
        for l in txt.lines() {
            let v: Vec<f64> = l.split(',').map(|s| s.parse().unwrap()).collect();
            if v[0] == 1.0 { yte.push(v[1]); xte.push(v[2..].to_vec()) } else { ytr.push(v[1]); xtr.push(v[2..].to_vec()) }
        }
        let mut pre = Preprocessor::new(PreprocessorConfig::default());
        let ztr = pre.fit_transform(&xtr).unwrap(); let zte = pre.transform(&xte).unwrap();
        let p = ztr[0].len();
        let (a, b) = (DenseMatrix::from_2d_vec(&ztr), DenseMatrix::from_2d_vec(&zte));
        let cfgs: Vec<(String, Option<usize>, usize)> = vec![
            ("sqrt".into(), None, 1), ("p/3".into(), Some((p / 3).max(1)), 1), ("p".into(), Some(p), 1), ("p/3,leaf3".into(), Some((p / 3).max(1)), 3), ("p/3,leaf5".into(), Some((p / 3).max(1)), 5),
        ];
        for (nm, m, leaf) in cfgs {
            let s = if is_clf {
                let (yt, ye): (Vec<u32>, Vec<u32>) = (ytr.iter().map(|v| *v as u32).collect(), yte.iter().map(|v| *v as u32).collect());
                let mut pr = RandomForestClassifierParameters::default(); pr.n_trees = 300; pr.m = m; pr.min_samples_leaf = leaf; pr.seed = 42;
                metrics::balanced_accuracy(&ye, &RandomForestClassifier::fit(&a, &yt, pr).unwrap().predict(&b).unwrap(), 2)
            } else {
                let mut pr = RandomForestRegressorParameters::default(); pr.n_trees = 300; pr.m = m; pr.min_samples_leaf = leaf; pr.seed = 42;
                metrics::r2(&yte, &RandomForestRegressor::fit(&a, &ytr, pr).unwrap().predict(&b).unwrap())
            };
            acc.entry((format!("{}:{}", &name[..3], scen.split_once('_').unwrap().1), nm)).or_default().push(s);
        }
    }
    for ((sc, nm), v) in acc { println!("{sc:28} {nm:10} {:.3}", v.iter().sum::<f64>() / v.len() as f64); }
}
