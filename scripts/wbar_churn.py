#!/usr/bin/env python3
"""
wbar_churn.py  —  per-model W-bar and churn at packet 8, sustained, canonical-65.

W-bar (per model): mean over seeds of the fraction of that model's TOTAL |SHAP|
importance mass that sits on UNSETTLED features (availability A_8 < GATE).
"Importance mass on features that haven't settled by the decision packet."

Churn (per model, at a given top-K): number of DISTINCT features that ever appear
in the model's top-K across the 5 seeds. A stable ranking touches exactly K
features (churn=K); every extra feature above K is churn. Reported for K=6
(the gate size) and as the raw count.

Everything is held identical to the sweep: same flows, same per-seed split, same
sustained population, same DELTA/GATE, same SHAP background (2000-row train sample,
same seed). Only the classifier moves.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd, shap
from sklearn.model_selection import train_test_split
from feature_set import canonical_features
from sweep_cross_classifier import make_model, shap_importance, need_label_encoding, merge_attempted

PKT, SEEDS = 8, [42,43,44,45,46]
MIN_SUS, DELTA, GATE, TOPK = 16, 0.05, 0.50, 6
MODELS = ["lightgbm","random_forest","xgboost"]

Xf = pd.read_parquet("shap_X.parquet").reset_index(drop=True)
y  = merge_attempted(pd.read_parquet("shap_y.parquet")["Label"]).reset_index(drop=True)
COLS = canonical_features(Xf.columns)
totp = (Xf["tot_fwd_pkts"]+Xf["tot_bwd_pkts"]).values
sus = totp >= MIN_SUS
cache = pickle.load(open("snap_cache.pkl","rb"))
CC = cache["cols"]; M, reached = cache["snaps"][PKT]
idx = [CC.index(f) for f in COLS]
Xfull = Xf[COLS].values.astype(np.float32); Msnap = M[:,idx].astype(np.float32)
classes = sorted(y.unique()); y2c = {c:i for i,c in enumerate(classes)}

rows_w, topk_sets = [], {m:[] for m in MODELS}
imp_rows = []
for KIND in MODELS:
    enc = need_label_encoding(KIND)
    yc = (lambda a: np.array([y2c[v] for v in a])) if enc else (lambda a: a)
    for SEED in SEEDS:
        tr, te = train_test_split(np.arange(len(Xf)), test_size=0.40, random_state=SEED, stratify=y)
        tr_sus = tr[sus[tr] & reached[tr]]
        # availability at packet 8 (identical across models)
        Xk=Msnap[tr_sus]; Xfin=Xfull[tr_sus]; den=np.maximum(np.abs(Xfin),1.0)
        av = pd.Series(((np.abs(Xk-Xfin)/den)<=DELTA).mean(0), index=COLS)
        settled = av >= GATE
        # model's own SHAP on FULL train values (deployment ranking source)
        Xdf = pd.DataFrame(Xfull[tr], columns=COLS)
        mdl = make_model(KIND,SEED).fit(Xdf, yc(y.values[tr]))
        imp = shap_importance(mdl, Xdf, SEED)            # mean|SHAP| per feature
        imp = imp.reindex(COLS).fillna(0.0)
        total = imp.sum()
        w_unsettled = imp[~settled].sum() / total if total>0 else np.nan
        rows_w.append(dict(classifier=KIND, seed=SEED, W_bar=round(w_unsettled,4),
                           n_unsettled=int((~settled).sum()), n_settled=int(settled.sum())))
        topk_sets[KIND].append(set(imp.sort_values(ascending=False).head(TOPK).index))

# W-bar summary
W = pd.DataFrame(rows_w)
Wsum = W.groupby("classifier")["W_bar"].agg(["mean","std"]).round(4)

# churn: distinct features ever in top-K across seeds, and per-seed jaccard stability
def churn_stats(sets):
    union = set().union(*sets); inter = set.intersection(*sets)
    return len(union), len(inter)
churn_rows=[]
for KIND in MODELS:
    u,i = churn_stats(topk_sets[KIND])
    churn_rows.append(dict(classifier=KIND, topK=TOPK, distinct_in_topK=u,
                           stable_core=i, churn_extra=u-TOPK))
C = pd.DataFrame(churn_rows)

W.to_csv("/mnt/user-data/outputs/wbar_per_seed_pkt8.csv", index=False)
C.to_csv("/mnt/user-data/outputs/churn_per_model_pkt8.csv", index=False)

print("=== W-bar (importance mass on UNSETTLED features, A_8<0.5) ===")
print("per-seed rows -> wbar_per_seed_pkt8.csv")
print(Wsum.to_string()); print()
print("=== churn (top-6 selection across 5 seeds) ===")
print(C.to_string(index=False))
print("\nchurn reading: distinct_in_topK = how many different features EVER enter top-6;")
print("stable_core = features in top-6 for ALL 5 seeds; churn_extra = distinct - 6.")
