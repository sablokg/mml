//! Ridge regression in dual (kernel) form with closed-form leave-one-out
//! selection of the penalty. Works for any n and p, unlike smartcore's Ridge
//! which needs n >= p.

use crate::error::{MicrobiomeError, Result};

/*
Gaurav Sablok
gsablok@proton.me
*/

#[derive(Debug, Clone)]
pub struct Ridge {
    w: Vec<f64>,
    x_mean: Vec<f64>,
    y_mean: f64,
    pub alpha: f64,
    pub loo_mse: f64,
}

/// In-place Cholesky of an SPD n x n row-major matrix; returns lower factor.
fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 1e-12 {
                    return None;
                }
                l[i * n + i] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    Some(l)
}

fn chol_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    x
}

impl Ridge {
    pub fn fit(x: &[Vec<f64>], y: &[f64]) -> Result<Self> {
        let (n, p) = (x.len(), x.first().map(|r| r.len()).unwrap_or(0));
        if n < 3 || p == 0 || y.len() != n {
            return Err(MicrobiomeError::InvalidInput(
                "ridge needs >= 3 samples and matching targets".into(),
            ));
        }
        let x_mean: Vec<f64> = (0..p)
            .map(|j| x.iter().map(|r| r[j]).sum::<f64>() / n as f64)
            .collect();
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let xc: Vec<Vec<f64>> = x
            .iter()
            .map(|r| r.iter().zip(&x_mean).map(|(v, m)| v - m).collect())
            .collect();
        let yc: Vec<f64> = y.iter().map(|v| v - y_mean).collect();
        let mut k = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let d: f64 = xc[i].iter().zip(&xc[j]).map(|(a, b)| a * b).sum();
                k[i * n + j] = d;
                k[j * n + i] = d;
            }
        }
        let scale = ((0..n).map(|i| k[i * n + i]).sum::<f64>() / n as f64).max(1e-12);
        let mut best: Option<(f64, f64, Vec<f64>)> = None; // (loo_mse, alpha, dual coef)
        for mult in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0] {
            let alpha = scale * mult;
            let mut a = k.clone();
            for i in 0..n {
                a[i * n + i] += alpha;
            }
            let Some(l) = cholesky(&a, n) else { continue };
            let coef = chol_solve(&l, n, &yc);
            // diag of (K + aI)^-1 via solving for unit vectors; H_ii = 1 - alpha * inv_ii
            let mut mse = 0.0;
            for i in 0..n {
                let mut e = vec![0.0; n];
                e[i] = 1.0;
                // The intercept is unpenalised: remove the constant direction (null
                // space of the centred kernel, eigenvalue alpha of A) from diag(A^-1).
                let inv_ii = chol_solve(&l, n, &e)[i] - 1.0 / (n as f64 * alpha);
                // LOO residual for kernel ridge: coef_i / inv_ii
                let r = coef[i] / inv_ii;
                mse += r * r;
            }
            mse /= n as f64;
            if best.as_ref().map_or(true, |b| mse < b.0) {
                best = Some((mse, alpha, coef));
            }
        }
        let (loo_mse, alpha, coef) = best.ok_or_else(|| {
            MicrobiomeError::Model("ridge: kernel matrix not positive definite".into())
        })?;
        let mut w = vec![0.0; p];
        for i in 0..n {
            for j in 0..p {
                w[j] += coef[i] * xc[i][j];
            }
        }
        Ok(Self {
            w,
            x_mean,
            y_mean,
            alpha,
            loo_mse,
        })
    }

    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|r| {
                self.y_mean
                    + r.iter()
                        .zip(&self.x_mean)
                        .zip(&self.w)
                        .map(|((v, m), w)| (v - m) * w)
                        .sum::<f64>()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn noise_target_is_not_interpolated() {
        // y independent of x with p > n: honest LOO error must be about var(y).
        let (n, p) = (40, 120);
        let mut s = 7u64;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        let x: Vec<Vec<f64>> = (0..n).map(|_| (0..p).map(|_| rnd()).collect()).collect();
        let y: Vec<f64> = (0..n).map(|_| rnd()).collect();
        let m = Ridge::fit(&x, &y).unwrap();
        let mean = y.iter().sum::<f64>() / n as f64;
        let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        assert!(m.loo_mse > 0.7 * var, "LOO {} vs var {}", m.loo_mse, var);
    }

    #[test]
    fn recovers_linear_signal_when_p_exceeds_n() {
        let (n, p) = (30, 80);
        let mut s = 1u64;
        let mut rnd = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
        };
        let x: Vec<Vec<f64>> = (0..n).map(|_| (0..p).map(|_| rnd()).collect()).collect();
        let y: Vec<f64> = x.iter().map(|r| 3.0 * r[0] - 2.0 * r[1] + 1.0).collect();
        let m = Ridge::fit(&x, &y).unwrap();
        let pred = m.predict(&x);
        let sse: f64 = pred.iter().zip(&y).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(sse / (n as f64) < 0.05, "mse too large");
        assert!(m.loo_mse.is_finite());
    }
}
