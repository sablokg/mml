"""Build benchmark tables from the public low-sample-size metagenomic sets (metageomic_datasets repo).
usage: python3 prep_metagenomic.py <metageomic_datasets dir> [outdir=data]
Output: <outdir>/<name>.tsv  (first column sample id, last column __label__ = control | disease, other columns = relative abundance)
and <outdir>/datasets_summary.csv. All tasks are binary (disease vs control)."""
import sys, os, numpy as np, pandas as pd
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "data"
os.makedirs(out, exist_ok=True)
rows = []

def uniq(names):
    seen = {}; res = []
    for n in names:
        n = str(n).strip()
        if n in seen: seen[n] += 1; n = f"{n}__dup{seen[n]}"
        else: seen[n] = 0
        res.append(n)
    return res

def save(name, X, y, group, source, positive, negative, note=""):
    """X: samples x taxa DataFrame; y: Series of 0/1 (1 = disease) aligned to X.index."""
    keep = ~y.isna(); X = X[keep.values]; y = y[keep.values].astype(int)
    X = X.loc[:, (X > 0).any(axis=0)]                      # drop all-zero taxa
    X = X.copy(); X.index.name = "sample"; X.columns = uniq(X.columns)
    X["__label__"] = np.where(y.values == 1, "disease", "control")
    X.to_csv(os.path.join(out, f"{name}.tsv"), sep="\t")
    rows.append(dict(dataset=name, group=group, source=source, n=len(y), n_disease=int(y.sum()), n_control=int((1 - y).sum()),
                     n_taxa=X.shape[1] - 1, disease_class=positive, control_class=negative, note=note))
    print(f"{name:22s} n={len(y):4d} ({int(y.sum())}/{int((1-y).sum())}) taxa={X.shape[1]-1}")

# ---- Disease-sets-1 (PopPhy-CNN / Meta-Signer): taxa x samples TSV (no header), labels.txt one label per sample
d1 = {"Cirrhosis": ("cirrhosis", "n"), "IBD": ("ibd", "n"), "Obesity": ("obesity", "leaness"), "T2D": ("t2d", "n"), "CRC": ("cancer", "n")}
for ds, (pos, neg) in d1.items():
    p = os.path.join(src, "Disease-sets-1", ds)
    A = pd.read_csv(os.path.join(p, "abundance.tsv"), sep="\t", header=None, index_col=0)
    lab = pd.Series([l.strip() for l in open(os.path.join(p, "labels.txt")) if l.strip()])
    assert A.shape[1] == len(lab), (ds, A.shape, len(lab))
    A.columns = range(A.shape[1]); X = A.T; X.index = [f"{ds}_{i}" for i in range(len(lab))]
    y = pd.Series(np.where(lab == pos, 1, np.where(lab == neg, 0, np.nan)), index=X.index)
    note = "adenoma samples (n=39) excluded: cancer vs healthy" if ds == "CRC" else ""
    save(f"DS1_{ds}", X, y, "Disease-sets-1", "Reiman et al. (PopPhy-CNN / Meta-Signer)", pos, neg, note)

# ---- Disease-sets-2 (Met2Img): taxa x samples CSV with header, *_y.csv columns id,x (1 = disease)
d2 = {"Cirrhosis": "cir", "IBD": "ibd", "Obesity": "obe", "T2D": "t2d", "WT2D": "wt2d", "CRC": "col"}
for ds, k in d2.items():
    p = os.path.join(src, "Disease-sets-2")
    A = pd.read_csv(os.path.join(p, f"{k}phy_x.csv"), index_col=0)
    yy = pd.read_csv(os.path.join(p, f"{k}phy_y.csv")).sort_values("id")
    assert A.shape[1] == len(yy), (ds, A.shape, len(yy))
    X = A.T; X.index = [f"{ds}_{i}" for i in range(X.shape[0])]
    y = pd.Series(yy["x"].values.astype(float), index=X.index)
    save(f"DS2_{ds}", X, y, "Disease-sets-2", "Nguyen et al. (Met2Img)", "case", "control")

# ---- CRC-Nation (Wirbel et al. 2019): one task per country, CRC vs CTR
X = pd.read_excel(os.path.join(src, "CRC-Nation", "dfx.xlsx"), index_col=0)
M = pd.read_excel(os.path.join(src, "CRC-Nation", "dfy.xlsx"), sheet_name="sample_info", index_col=0).loc[X.index]
for c, code in [("AUS", "AUS"), ("CHN", "CHI"), ("DEU", "GER"), ("FRA", "FRA"), ("USA", "USA")]:   # metadata uses CHI / GER
    m = M[M.Country == code]
    save(f"CRCNation_{c}", X.loc[m.index].astype(float), pd.Series((m.Group == "CRC").astype(float).values, index=m.index),
         "CRC-Nation", "Wirbel et al. 2019", "CRC", "CTR")

# ---- CRC-Stage (Yachida et al. 2019): each stage vs the same 127 healthy controls
X = pd.read_csv(os.path.join(src, "CRC-Stage", "dfx.csv"), index_col=0)
M = pd.read_csv(os.path.join(src, "CRC-Stage", "dfy.csv"), index_col=0).loc[X.index]
print("CRC-Stage groups:", M.Group.value_counts().to_dict())
for g, tag in [("MP", "MP"), ("Stage_0", "S0"), ("Stage_I_II", "SI-II"), ("Stage_III_IV", "SIII-IV")]:
    m = M[M.Group.isin([g, "Healthy"])]
    assert (m.Group == g).sum() > 0, g
    save(f"CRCStage_{tag}", X.loc[m.index].astype(float), pd.Series((m.Group == g).astype(float).values, index=m.index),
         "CRC-Stage", "Yachida et al. 2019", g, "Healthy")

pd.DataFrame(rows).to_csv(os.path.join(out, "datasets_summary.csv"), index=False)
