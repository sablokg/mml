use microbiome_ml::cv::Lcg;
use microbiome_ml::prelude::*;

/// 2-group cohort: taxa 0..4 enriched in group 1; the rest are noise.
fn cohort(n: usize, p: usize, seed: u64) -> (Dataset, Vec<String>) {
    let mut r = Lcg::new(seed);
    let mut x = vec![];
    let mut y = vec![];
    for i in 0..n {
        let g = i % 2;
        let row: Vec<f64> = (0..p)
            .map(|j| {
                let base = 5.0 + (r.next_u64() % 30) as f64;
                let boost = if j < 4 && g == 1 { 6.0 } else { 1.0 };
                if r.next_u64() % 6 == 0 { 0.0 } else { (base * boost).floor() }
            })
            .collect();
        x.push(row);
        y.push(if g == 1 { "case".to_string() } else { "control".to_string() });
    }
    let names: Vec<String> = (0..p).map(|j| format!("taxon_{j}")).collect();
    (Dataset::new(x, Some(names), None).unwrap(), y)
}

#[test]
fn permutation_importance_recovers_enriched_taxa_by_name() {
    let (train, ytr) = cohort(90, 30, 1);
    let (test, yte) = cohort(60, 30, 2);
    let mut auto = AutoClassifier::default();
    auto.transforms = vec![Transform::Clr];
    auto.models = vec![ClassifierKind::RandomForest];
    let fit = auto.fit(&train, &ytr).unwrap();
    let imp = fit.taxon_importance(&test.x, &yte, 5, 7).unwrap();
    let top4: Vec<&str> = imp.iter().take(4).map(|t| t.taxon.as_str()).collect();
    let hits = top4.iter().filter(|n| ["taxon_0", "taxon_1", "taxon_2", "taxon_3"].contains(n)).count();
    assert!(hits >= 3, "top taxa were {top4:?}");
    println!("{}", format_importance(&imp, 5));
}

#[test]
fn grid_search_adds_tuned_candidates_and_never_hurts_cv_score() {
    let (train, y) = cohort(80, 20, 3);
    let base = |g| {
        let mut a = AutoClassifier::default().with_grid(g);
        a.transforms = vec![Transform::Clr];
        a.models = vec![ClassifierKind::LogisticRegression, ClassifierKind::Knn];
        a.refine_top = 2;
        a
    };
    let off = base(GridLevel::Off).fit(&train, &y).unwrap();
    let on = base(GridLevel::Small).fit(&train, &y).unwrap();
    assert!(on.leaderboard.len() > off.leaderboard.len());
    assert!(on.best.score >= off.best.score - 1e-12);
    assert!(on.leaderboard.iter().any(|c| !c.params.is_default()));
    assert!(on.report().contains("alpha=") || on.report().contains("k="));
}

#[test]
fn phylogeny_adds_ilr_and_clade_transforms_to_the_search() {
    let (train, y) = cohort(60, 8, 4);
    let tree = PhyloTree::from_newick("(((taxon_0:1,taxon_1:1):1,(taxon_2:1,taxon_3:1):1):1,((taxon_4:1,taxon_5:1):1,(taxon_6:1,taxon_7:1):1):1);").unwrap();
    let mut a = AutoClassifier::default().with_phylogeny(tree, &train.feature_names);
    a.models = vec![ClassifierKind::LogisticRegression];
    a.transforms.retain(|t| matches!(t, Transform::Clr | Transform::Ilr | Transform::CladeClr));
    a.preprocess.min_prevalence = 0.0;
    let fit = a.fit(&train, &y).unwrap();
    let seen: Vec<_> = fit.leaderboard.iter().map(|c| c.transform).collect();
    assert!(seen.contains(&Transform::Ilr) && seen.contains(&Transform::CladeClr), "{seen:?}; failures {:?}", fit.failures);
    let preds = fit.predict(&train.x).unwrap();
    assert_eq!(preds.len(), 60);
}

#[test]
fn data_driven_ilr_runs_without_a_tree() {
    let (train, y) = cohort(60, 12, 5);
    let mut a = AutoClassifier::default();
    a.transforms = vec![Transform::Ilr];
    a.models = vec![ClassifierKind::LogisticRegression];
    a.preprocess.min_prevalence = 0.0;
    let fit = a.fit(&train, &y).unwrap();
    assert_eq!(fit.preprocessor.output_names(&fit.feature_names).len(), 11);
}

#[cfg(feature = "xgboost")]
#[test]
fn xgboost_via_smelt_classifies_regresses_and_reports_native_importance() {
    let (train, y) = cohort(100, 25, 6);
    let mut a = AutoClassifier::default().with_grid(GridLevel::Small);
    a.transforms = vec![Transform::Clr];
    a.models = vec![ClassifierKind::XgBoost];
    a.refine_top = 1;
    let fit = a.fit(&train, &y).unwrap();
    assert!(fit.best.score > 0.7, "xgb cv bal.acc {}", fit.best.score);
    let nat = fit.native_importance().expect("xgboost provides native importance");
    assert!(nat.iter().take(3).any(|(n, _)| ["taxon_0", "taxon_1", "taxon_2", "taxon_3"].contains(&n.as_str())), "{nat:?}");

    let target: Vec<f64> = train.x.iter().map(|r| r[0] / r.iter().sum::<f64>() * 100.0).collect();
    let mut r = AutoRegressor::default();
    r.transforms = vec![Transform::Clr];
    r.models = vec![RegressorKind::XgBoost];
    let rf = r.fit(&train, &target).unwrap();
    assert!(rf.predict(&train.x).unwrap().len() == 100);
}
