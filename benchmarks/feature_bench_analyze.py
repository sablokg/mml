import pandas as pd, numpy as np
from scipy.stats import wilcoxon
pd.set_option("display.width",200)
d=pd.read_csv('feature_bench_results.csv')
ch=d[d.method.str.contains(':choice')].copy()
d=d[~d.method.str.contains(':choice')]
print("reps per scenario:", d.groupby('scenario').rep.nunique().to_dict())
fixed=d[~d.method.str.startswith('importance')]
t=fixed.pivot_table(index='method',columns='scenario',values='value',aggfunc=['mean','std']).round(3)
print("\nHeld-out balanced accuracy (n=120,p=100, mean/sd over reps)"); print(t)
print("\nmean secs per fit"); print(fixed.pivot_table(index='method',columns='scenario',values='secs',aggfunc='mean').round(2))
def paired(sc,a,b):
    p=fixed[fixed.scenario==sc].pivot_table(index='rep',columns='method',values='value')
    dl=(p[a]-p[b]).dropna()
    try: pv=wilcoxon(dl).pvalue
    except Exception: pv=float('nan')
    return f"{sc:9s} {a} - {b}: {dl.mean():+.3f} (W/T/L {int((dl>0).sum())}/{int((dl==0).sum())}/{int((dl<0).sum())}, p={pv:.3f})"
print("\nPaired contrasts")
for sc in ['clade','scattered']:
    for a,b in [('ILRphylo+LR','CLR+LR'),('ILRdata+LR','CLR+LR'),('CladeCLR+LR','CLR+LR'),('ILRphylo+RF','CLR+RF'),('ILRdata+RF','CLR+RF'),('CLR+XGB','CLR+RF'),('ILRphylo+XGB','CLR+XGB'),('AUTO_grid','AUTO_base'),('AUTO_phylo','AUTO_base')]:
        print(paired(sc,a,b))
print("\nAUTO choices"); ch['c']=ch.method; print(ch.groupby(['scenario','c']).size().to_string())
imp=d[d.method.str.startswith('importance')]
print("\nTop-k precision of importance rankings (k=6 true signal taxa; chance = 6/kept ~0.06)")
print(imp.pivot_table(index='method',columns='scenario',values='value',aggfunc=['mean','std']).round(3))
