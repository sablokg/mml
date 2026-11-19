# scikit-learn reference on the identical folds written by zhu_bench (CLR + LR, CLR + RF). Run from real_benchmark/.
import sys, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
level = sys.argv[1]
t = pd.read_csv(f"data/{level}.tsv", sep="\t", index_col=0)
y = (t.pop("__label__") == "SCZ").astype(int).values; X = t.values.astype(float)
folds = pd.read_csv(f"results/folds_{level}.csv")
def prep(Xtr, Xte, min_prev=0.1):
    keep = (Xtr > 0).mean(0) >= min_prev
    def tss(M):
        M = M[:, keep]; s = M.sum(1, keepdims=True); s[s == 0] = 1; return M / s
    A, B = tss(Xtr), tss(Xte); pc = A[A > 0].min() / 2
    clr = lambda R: (lambda L: L - L.mean(1, keepdims=True))(np.log(R + pc))
    return clr(A), clr(B)
rows = []
for _, r in folds.iterrows():
    te = np.array(str(r.test_idx).split(), dtype=int); tr = np.setdiff1d(np.arange(len(y)), te)
    Ztr, Zte = prep(X[tr], X[te]); sc = StandardScaler().fit(Ztr)
    def run(name, fn):
        t0 = time.time(); s = fn(); rows.append(dict(rep=r.rep, fold=r.fold, method=name, bal_acc=s, secs=time.time() - t0))
    run("sk_CLR+LR", lambda: balanced_accuracy_score(y[te], LogisticRegression(max_iter=2000).fit(sc.transform(Ztr), y[tr]).predict(sc.transform(Zte))))
    run("sk_CLR+RF", lambda: balanced_accuracy_score(y[te], RandomForestClassifier(100, random_state=0, n_jobs=1).fit(Ztr, y[tr]).predict(Zte)))
pd.DataFrame(rows).to_csv(f"results/sk_{level}.csv", index=False); print("ok", level)
