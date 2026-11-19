use microbiome_ml::cv::Lcg;
use microbiome_ml::prelude::*;

/// Synthetic data: 2 groups differing in a few taxa, plus a continuous target.
fn main() -> Result<()> {
    let mut rng = Lcg::new(7);
    let (n, p) = (80, 40);
    let mut x = vec![];
    let (mut labels, mut target) = (vec![], vec![]);
    for i in 0..n {
        let g = i % 2;
        let row: Vec<f64> = (0..p)
            .map(|j| {
                let base = 5.0 + (rng.next_u64() % 40) as f64;
                let boost = if j < 4 && g == 1 { 6.0 } else { 1.0 };
                if rng.next_u64() % 5 == 0 { 0.0 } else { (base * boost).floor() }
            })
            .collect();
        target.push(row[0] / row.iter().sum::<f64>() * 100.0);
        x.push(row);
        labels.push(if g == 1 { "case".to_string() } else { "control".to_string() });
    }
    let data = Dataset::new(x, None, None)?;

    let clf = AutoClassifier::default().fit(&data, &labels)?;
    println!("{}\n", clf.report());
    println!("first predictions: {:?}\n", &clf.predict(&data.x)?[..4]);

    let reg = AutoRegressor::default().fit(&data, &target)?;
    println!("{}", reg.report());
    Ok(())
}
