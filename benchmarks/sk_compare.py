import glob, os, re, sys, time, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, r2_score

def prep(Xtr, Xte, min_prev=0.1):
    keep = (Xtr > 0).mean(0) >= min_prev
    def tss(X):
        X = X[:, keep]; s = X.sum(1, keepdims=True); s[s == 0] = 1; return X / s
    A, B = tss(Xtr), tss(Xte)
    pc = A[A > 0].min() / 2
    def clr(R):
        L = np.log(R + pc); return L - L.mean(1, keepdims=True)
    return clr(A), clr(B)

rows = []
for f in sorted(glob.glob(sys.argv[1] + "/data/*.csv")):
    m = re.match(r"(clf|reg)_n(\d+)_p(\d+)_e([\d.]+)_r(\d+)", os.path.basename(f))
    task, n, p, eff, rep = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5))
    d = np.loadtxt(f, delimiter=",")
    te = d[:, 0].astype(bool); y = d[:, 1]; X = d[:, 2:]
    Ztr, Zte = prep(X[~te], X[te])
    sc = StandardScaler().fit(Ztr); Str, Ste = sc.transform(Ztr), sc.transform(Zte)
    ytr, yte = y[~te], y[te]
    def run(name, fn):
        t = time.time(); s = fn(); rows.append(dict(task=task, n=n, p=p, effect=eff, rep=rep, method=name, score=s, fit_s=time.time() - t))
    if task == "clf":
        run("sk_Clr+RandomForest", lambda: balanced_accuracy_score(yte, RandomForestClassifier(300, random_state=0, n_jobs=1).fit(Ztr, ytr.astype(int)).predict(Zte)))
        run("sk_Clr+LogisticRegression", lambda: balanced_accuracy_score(yte, LogisticRegression(max_iter=2000).fit(Str, ytr.astype(int)).predict(Ste)))
    else:
        run("sk_Clr+RandomForest", lambda: r2_score(yte, RandomForestRegressor(300, random_state=0, n_jobs=1).fit(Ztr, ytr).predict(Zte)))
        run("sk_Clr+Ridge", lambda: r2_score(yte, RidgeCV(alphas=np.logspace(-3, 5, 17)).fit(Str, ytr).predict(Ste)))
pd.DataFrame(rows).to_csv(sys.argv[1] + "/sklearn.csv", index=False)
print("ok", len(rows))
