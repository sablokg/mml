"""Summarise results/: mean (SD) balanced accuracy over repeated 5-fold CV, paired Wilcoxon vs CLR+LR. Run from real_benchmark/.
usage: python3 analyze.py [dataset ...]   (default: every results/res_<name>.csv that has a matching sk_<name>.csv)
Per dataset: results/summary_<name>.csv, results/auto_choices_<name>.csv. Across datasets: results/benchmark_all_long.csv,
results/benchmark_matrix_mean.csv (dataset x method), results/benchmark_ranks.csv (mean rank, wins, mean difference vs CLR+LR)."""
import sys, glob, os, pandas as pd, numpy as np
from scipy.stats import wilcoxon, friedmanchisquare
names = sys.argv[1:] or sorted(os.path.basename(f)[4:-4] for f in glob.glob("results/res_*.csv") if os.path.exists(f.replace("res_", "sk_")))
long = []
for lvl in names:
    r = pd.read_csv(f"results/res_{lvl}.csv"); sk = pd.read_csv(f"results/sk_{lvl}.csv")
    choice = r[r.method.str.startswith("AUTO:choice")].method.str.replace("AUTO:choice=", "").value_counts()
    a = pd.concat([r[~r.method.str.startswith("AUTO:choice")], sk])
    g = a.groupby("method").agg(mean=("bal_acc", "mean"), sd=("bal_acc", "std"), folds=("bal_acc", "count"), secs_per_fit=("secs", "mean"))
    na = list(g.index[g.folds == 0]); g = g[g.folds > 0]                      # methods that failed on every fold (e.g. data-driven ILR above 2000 taxa)
    if na: print(f"[{lvl}] no scores for: {na}")
    a = a[a.method.isin(g.index)]
    piv = a.pivot_table(index=["rep", "fold"], columns="method", values="bal_acc")
    g["diff_vs_CLR+LR"] = [(piv[m] - piv["CLR+LR"]).mean() for m in g.index]
    def pv(m):
        d = (piv[m] - piv["CLR+LR"]).dropna()
        return float("nan") if m == "CLR+LR" or (d == 0).all() else wilcoxon(d).pvalue
    g["p_vs_CLR+LR"] = [pv(m) for m in g.index]
    g = g.sort_values("mean", ascending=False).round(4)
    g.to_csv(f"results/summary_{lvl}.csv"); choice.to_csv(f"results/auto_choices_{lvl}.csv", header=["folds"])
    print(f"\n== {lvl} ==\n{g.to_string()}\n\nAUTO choices (outer folds):\n{choice.to_string()}")
    long.append(g.reset_index().assign(dataset=lvl))
new = [n for n in names if n not in ("genus", "species")]                  # Zhu genus/species are two levels of one cohort: not pooled
if len(new) > 2:
    L = pd.concat([l for l, n in zip(long, names) if n in new]); L = L[L.method != "AUTO"]   # AUTO was not run on these datasets
    L.to_csv("results/benchmark_all_long.csv", index=False)
    M = L.pivot(index="dataset", columns="method", values="mean"); M.round(4).to_csv("results/benchmark_matrix_mean.csv")
    full_cols = [c for c in M.columns if M[c].notna().all()]                   # methods that produced scores on every dataset
    R = M[full_cols].rank(axis=1, ascending=False)                             # rank 1 = best on that dataset
    D = L.pivot(index="dataset", columns="method", values="diff_vs_CLR+LR")
    S = pd.DataFrame({"datasets": M.notna().sum(), "mean_bal_acc": M.mean(), "mean_rank": R.mean(), "wins": (R == 1).sum(),
                      "mean_diff_vs_CLR+LR": D.mean(), "datasets_better_than_CLR+LR": (D > 0).sum(), "datasets_worse_than_CLR+LR": (D < 0).sum(),
                      "secs_per_fit": L.groupby("method").secs_per_fit.mean()}).sort_values("mean_rank").round(4)
    S.to_csv("results/benchmark_ranks.csv")
    print(f"\n== across {len(M)} datasets (ranks among methods with scores everywhere: {full_cols}) ==\n{S.to_string()}")
    if len(M) > 3 and len(full_cols) > 2:
        print("Friedman test over", len(full_cols), "methods x", len(M), "datasets: p =", friedmanchisquare(*[M[c] for c in full_cols]).pvalue)
        # pairwise, paired over datasets (dataset-level means), vs CLR+LR
        for c in full_cols:
            if c != "CLR+LR": print(f"  Wilcoxon over datasets, {c} vs CLR+LR: diff {np.mean(M[c]-M['CLR+LR']):+.4f}, p = {wilcoxon(M[c]-M['CLR+LR']).pvalue:.3f}")
