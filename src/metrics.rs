/*
Gaurav Sablok
gsablok@proton.me
*/

pub fn accuracy(y: &[u32], p: &[u32]) -> f64 {
    y.iter().zip(p).filter(|(a, b)| a == b).count() as f64 / y.len().max(1) as f64
}

/// Mean per-class recall; robust to the class imbalance common in cohorts.
pub fn balanced_accuracy(y: &[u32], p: &[u32], n_classes: usize) -> f64 {
    let (mut hit, mut tot) = (vec![0usize; n_classes], vec![0usize; n_classes]);
    for (a, b) in y.iter().zip(p) {
        tot[*a as usize] += 1;
        if a == b {
            hit[*a as usize] += 1;
        }
    }
    let present: Vec<f64> = (0..n_classes)
        .filter(|&c| tot[c] > 0)
        .map(|c| hit[c] as f64 / tot[c] as f64)
        .collect();
    present.iter().sum::<f64>() / present.len().max(1) as f64
}

pub fn macro_f1(y: &[u32], p: &[u32], n_classes: usize) -> f64 {
    let mut f = Vec::new();
    for c in 0..n_classes as u32 {
        let tp = y
            .iter()
            .zip(p)
            .filter(|(a, b)| **a == c && **b == c)
            .count() as f64;
        let fp = y
            .iter()
            .zip(p)
            .filter(|(a, b)| **a != c && **b == c)
            .count() as f64;
        let fneg = y
            .iter()
            .zip(p)
            .filter(|(a, b)| **a == c && **b != c)
            .count() as f64;
        if tp + fneg == 0.0 {
            continue;
        }
        let d = 2.0 * tp + fp + fneg;
        f.push(if d > 0.0 { 2.0 * tp / d } else { 0.0 });
    }
    f.iter().sum::<f64>() / f.len().max(1) as f64
}

pub fn rmse(y: &[f64], p: &[f64]) -> f64 {
    (y.iter().zip(p).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / y.len().max(1) as f64).sqrt()
}

pub fn mae(y: &[f64], p: &[f64]) -> f64 {
    y.iter().zip(p).map(|(a, b)| (a - b).abs()).sum::<f64>() / y.len().max(1) as f64
}

pub fn r2(y: &[f64], p: &[f64]) -> f64 {
    let m = y.iter().sum::<f64>() / y.len().max(1) as f64;
    let ss_tot: f64 = y.iter().map(|a| (a - m).powi(2)).sum();
    let ss_res: f64 = y.iter().zip(p).map(|(a, b)| (a - b).powi(2)).sum();
    if ss_tot > 0.0 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    }
}
