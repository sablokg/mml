//! Deterministic (seeded) fold generation, no external RNG dependency.

/*
Gaurav Sablok
gsablok@proton.me
*/

pub struct Lcg(u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, (self.next_u64() % (i as u64 + 1)) as usize);
        }
    }
}

/// Returns `k` (train_idx, test_idx) pairs. Stratified on `strata` when given.
pub fn kfold(
    n: usize,
    k: usize,
    strata: Option<&[u32]>,
    seed: u64,
) -> Vec<(Vec<usize>, Vec<usize>)> {
    let k = k.clamp(2, n.max(2));
    let mut rng = Lcg::new(seed);
    let mut fold_of = vec![0usize; n];
    match strata {
        Some(s) => {
            let n_cls = s.iter().copied().max().map(|m| m as usize + 1).unwrap_or(0);
            let mut offset = 0;
            for c in 0..n_cls {
                let mut idx: Vec<usize> = (0..n).filter(|&i| s[i] as usize == c).collect();
                rng.shuffle(&mut idx);
                for (r, i) in idx.into_iter().enumerate() {
                    fold_of[i] = (r + offset) % k;
                }
                offset += 1;
            }
        }
        None => {
            let mut idx: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut idx);
            for (r, i) in idx.into_iter().enumerate() {
                fold_of[i] = r % k;
            }
        }
    }
    (0..k)
        .map(|f| {
            (
                (0..n).filter(|&i| fold_of[i] != f).collect(),
                (0..n).filter(|&i| fold_of[i] == f).collect(),
            )
        })
        .collect()
}

pub fn take<T: Clone>(v: &[T], idx: &[usize]) -> Vec<T> {
    idx.iter().map(|&i| v[i].clone()).collect()
}
