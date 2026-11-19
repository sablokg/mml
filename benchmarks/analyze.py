import pandas as pd, numpy as np
pd.set_option("display.width",200); pd.set_option("display.max_rows",200)
r = pd.read_csv("bench_final/results.csv"); s = pd.read_csv("bench_final/sklearn.csv")
r["scen"] = r.apply(lambda x: f"n={x.n},p={x.p},eff={x.effect}", axis=1)
r["auto"] = r.method.str.startswith("AUTO")
print("rows", len(r), "reps per scen", r.groupby("scen").rep.nunique().to_dict())
# 1. AUTO vs best-fixed, vs fixed defaults, per task/scenario
for task in ["clf","reg"]:
    d = r[r.task==task]
    print("\n==", task, "mean test score by scenario")
    piv = d.pivot_table(index="scen", columns="method", values="score", aggfunc="mean")
    cols = [c for c in piv.columns if c.startswith("AUTO")]
    d2 = d.copy(); d2["m"] = np.where(d2.auto,"AUTO",d2.method)
    t = d2.groupby(["scen","m"]).score.agg(["mean","std"]).reset_index()
    for sc in sorted(d.scen.unique(), key=lambda z: (int(z.split(',')[1][2:]), z)):
        tt = t[t.scen==sc].sort_values("mean", ascending=False)
        a = tt[tt.m=="AUTO"].iloc[0]; fixed = tt[tt.m!="AUTO"]
        print(f"{sc:24s} AUTO {a['mean']:.3f}±{a['std']:.3f} | best-fixed {fixed.iloc[0].m} {fixed.iloc[0]['mean']:.3f} | median-fixed {fixed['mean'].median():.3f} | Tss+RF {fixed[fixed.m=='Tss+RandomForest']['mean'].values[0]:.3f} | Clr+RF {fixed[fixed.m=='Clr+RandomForest']['mean'].values[0]:.3f}")
# 2. What does AUTO pick
print("\nAUTO choices"); print(r[r.auto].groupby(["task","method"]).size())
# 3. mean over all scenarios by transform / model (fixed)
f = r[~r.auto].copy(); f["transform"]=f.method.str.split("+").str[0]; f["model"]=f.method.str.split("+").str[1]
for task in ["clf","reg"]:
    print("\n", task, "transform x model mean score"); print(f[f.task==task].pivot_table(index="transform",columns="model",values="score",aggfunc="mean").round(3))
# 4. AUTO time
print("\nAUTO fit time"); print(r[r.auto].groupby(["task","scen"]).fit_s.agg(["mean","std"]).round(2))
# 5. sklearn comparison on the shared pipelines
s["scen"] = s.apply(lambda x: f"n={x.n},p={x.p},eff={x.effect}", axis=1)
m = {"sk_Clr+RandomForest":"Clr+RandomForest","sk_Clr+LogisticRegression":"Clr+LogisticRegression","sk_Clr+Ridge":"Clr+Ridge"}
s["method2"]=s.method.map(m)
j = s.merge(r[~r.auto], left_on=["task","scen","rep","method2"], right_on=["task","scen","rep","method"], suffixes=("_sk","_rs"))
print("\nsklearn vs Rust, per method"); 
g = j.groupby(["task","method2"]).agg(sk=("score_sk","mean"), rs=("score_rs","mean"), d=("score_rs", lambda x: 0), sk_t=("fit_s_sk","mean"), rs_t=("fit_s_rs","mean"))
j["diff"]=j.score_rs-j.score_sk
g["mean_diff"]=j.groupby(["task","method2"])["diff"].mean(); g["sd_diff"]=j.groupby(["task","method2"])["diff"].std(); g["speed_ratio"]=g.sk_t/g.rs_t
print(g.drop(columns="d").round(4))
print("\nper-scenario RF diff"); print(j[j.method2=="Clr+RandomForest"].groupby(["task","scen"])["diff"].mean().round(3))
sc = pd.read_csv("bench_final/scaling.csv"); print("\nscaling"); print(sc.pivot_table(index=["n","p"],columns="stage",values="secs").round(3).T)

# ---- paired statistics
from scipy.stats import wilcoxon
print("\n=== paired tests (per-dataset, pooled over scenarios) ===")
rr = r.copy(); rr["key"]=rr.task+"|"+rr.scen+"|"+rr.rep.astype(str)
def piv(task): 
    d=rr[rr.task==task]; return d.pivot_table(index="key",columns="method",values="score",aggfunc="first")
for task in ["clf","reg"]:
    P=piv(task); auto=P[[c for c in P.columns if c.startswith("AUTO")]].bfill(axis=1).iloc[:,0]
    fixed=P[[c for c in P.columns if not c.startswith("AUTO")]]
    top=fixed.mean().sort_values(ascending=False)
    print(f"\n{task}: top fixed pipelines by mean over {len(P)} datasets"); print(top.head(5).round(3))
    print(f"AUTO mean {auto.mean():.3f}; median fixed {fixed.mean().median():.3f}")
    for ref in [top.index[0], "Clr+RandomForest", "Tss+RandomForest"]:
        dlt=(auto-fixed[ref]).dropna(); 
        try: pv=wilcoxon(dlt).pvalue
        except Exception: pv=float("nan")
        print(f"AUTO - {ref}: mean diff {dlt.mean():+.3f}  (win/tie/loss {int((dlt>0).sum())}/{int((dlt==0).sum())}/{int((dlt<0).sum())}) Wilcoxon p={pv:.3g}")
    # transform contrasts within model family
    models = ["LogisticRegression","RandomForest"] if task=="clf" else ["ElasticNet","Ridge","RandomForest"]
    for m in models:
        for a,b in [("Clr","Tss"),("Clr","ArcsinSqrt"),("Clr","LogTss"),("Clr","None")]:
            if f"{a}+{m}" in fixed and f"{b}+{m}" in fixed:
                dlt=(fixed[f"{a}+{m}"]-fixed[f"{b}+{m}"]).dropna()
                try: pv=wilcoxon(dlt).pvalue
                except Exception: pv=float("nan")
                print(f"  {m:20s} {a} - {b}: {dlt.mean():+.3f} (p={pv:.3g})")
print("\nNaN (failed) rows by method:"); print(r[r.score.isna()].groupby(["task","method"]).size())
