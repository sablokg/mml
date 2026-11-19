"""Summarise results/: mean (SD) balanced accuracy over repeated 5-fold CV, paired Wilcoxon vs CLR+LR. Run from real_benchmark/."""
import pandas as pd
from scipy.stats import wilcoxon
for lvl in ["genus", "species"]:
    r = pd.read_csv(f"results/res_{lvl}.csv"); sk = pd.read_csv(f"results/sk_{lvl}.csv")
    choice = r[r.method.str.startswith("AUTO:choice")].method.str.replace("AUTO:choice=", "").value_counts()
    a = pd.concat([r[~r.method.str.startswith("AUTO:choice")], sk])
    g = a.groupby("method").agg(mean=("bal_acc", "mean"), sd=("bal_acc", "std"), folds=("bal_acc", "count"), secs_per_fit=("secs", "mean"))
    piv = a.pivot_table(index=["rep", "fold"], columns="method", values="bal_acc")
    g["diff_vs_CLR+LR"] = [(piv[m] - piv["CLR+LR"]).mean() for m in g.index]
    g["p_vs_CLR+LR"] = [wilcoxon(piv[m] - piv["CLR+LR"]).pvalue if m != "CLR+LR" else float("nan") for m in g.index]
    g = g.sort_values("mean", ascending=False).round(4)
    g.to_csv(f"results/summary_{lvl}.csv"); choice.to_csv(f"results/auto_choices_{lvl}.csv", header=["folds"])
    print(f"\n== {lvl} ==\n{g.to_string()}\n\nAUTO choices (outer folds):\n{choice.to_string()}")
