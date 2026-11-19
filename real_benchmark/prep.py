"""Build genus/species count tables from the Zhu et al. (2020) MetaPhlAn counts.
usage: python3 prep.py <dir with zhu_metaphlan_read_counts.txt and zhu_supplemental.tsv> [outdir]
Source files ship with the ML-for-microbiomics tutorial data. Output: <outdir>/{genus,species}.tsv
(first column sample id, last column __label__ = HC | SCZ, other columns raw counts)."""
import sys, os, pandas as pd
src = sys.argv[1]; out = sys.argv[2] if len(sys.argv) > 2 else "data"
os.makedirs(out, exist_ok=True)
df = pd.read_csv(os.path.join(src, "zhu_metaphlan_read_counts.txt"), sep="\t", header=1, index_col=0)
md = pd.read_csv(os.path.join(src, "zhu_supplemental.tsv"), skiprows=3, index_col=0, sep="\t").iloc[:, :-4]
ids = [c for c in df.columns if c in md.index]
df, md = df[ids], md.loc[ids]
def level(df, l):
    if l == "species": return df[df.index.str.contains("s__") & ~df.index.str.contains("t__")]
    return df[df.index.str.contains("g__") & ~df.index.str.contains("s__")]
for l in ["genus", "species"]:
    t = level(df, l).T
    t.index.name = "sample"
    t.columns = [c.split("|")[-1] for c in t.columns]
    t["__label__"] = md["Group"].values
    t.to_csv(os.path.join(out, f"{l}.tsv"), sep="\t")
    print(l, t.shape[0], "samples x", t.shape[1] - 1, "taxa")
