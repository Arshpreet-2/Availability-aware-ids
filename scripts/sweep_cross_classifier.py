#!/usr/bin/env python3
"""
sweep_cross_classifier.py
============================================================================
Cross-classifier robustness sweep. Same packet-8 setup as
sweep_one_packet_category.py, but run under a chosen CLASSIFIER so we can show
the deployment collapse + stability gap are not a LightGBM/SHAP artifact.

Fixed to the robustness-figure spec:
  * macro-F1 ONLY
  * per-seed rows (NO aggregation -- caller computes mean/SD)
  * both methods (availability, importance)
  * all three regimes (F/F, F/P, P/P)  [F/P is the one that matters]
  * K = 2 .. gate_size
  * CATEGORY pool (genuine+late INTERSECT canonical-65), sustained population
  * PER-MODEL selection: the importance ranking AND the availability gate order
    are computed from THIS classifier's own SHAP values (honest cross-model test:
    each model selects with its own importances).

CSV schema (exactly):
    classifier, regime, method, K, seed, macro_f1

USAGE
    python sweep_cross_classifier.py --classifier random_forest --seeds 42 43 44 45 46
    python sweep_cross_classifier.py --classifier xgboost       --seeds 42 43 44 45 46
    python sweep_cross_classifier.py -c random_forest -p 8 -s 42

Output: sweep_cross_<classifier>_pkt<P>_seeds<list>.csv
============================================================================
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, pickle
import numpy as np, pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from feature_set import canonical_features

MIN_SUS = 16
DELTA   = 0.05
GATE    = 0.50
SHAP_BG = 2000          # background sample for SHAP (train only)
REGIMES = ["F/F", "F/P", "P/P"]

def merge_attempted(s):
    return s.str.replace(" - Attempted", "", regex=False)

def make_model(kind, seed):
    if kind == "random_forest":
        return RandomForestClassifier(n_estimators=200, max_depth=None,
                                      class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=-1)
    if kind == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=6,
                                 subsample=0.9, colsample_bytree=0.9,
                                 tree_method="hist", random_state=seed,
                                 n_jobs=-1, eval_metric="mlogloss")
    if kind == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=250, learning_rate=0.1, num_leaves=31,
                                  class_weight="balanced", random_state=seed,
                                  n_jobs=-1, verbose=-1)
    raise ValueError(kind)

def shap_importance(model, Xdf, seed):
    """mean|SHAP| per feature, robust across shap output shapes; XGB/LGB use
    TreeExplainer, RF too (tree model)."""
    bg = Xdf.sample(n=min(SHAP_BG, len(Xdf)), random_state=seed)
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(bg)
    if isinstance(sv, list):            # list per class (RF, multiclass)
        impv = np.mean([np.abs(a).mean(0) for a in sv], 0)
    elif np.abs(sv).ndim == 3:          # (n, feat, class)
        impv = np.abs(sv).mean((0, 2))
    else:
        impv = np.abs(sv).mean(0)
    return pd.Series(impv, index=Xdf.columns)

def need_label_encoding(kind):
    return kind == "xgboost"   # xgboost wants 0..C-1 integer labels

def run(KIND, PKT, SEEDS):
    Xf = pd.read_parquet("shap_X.parquet").reset_index(drop=True)
    y  = merge_attempted(pd.read_parquet("shap_y.parquet")["Label"]).reset_index(drop=True)
    COLS_65 = canonical_features(Xf.columns)
    totp = (Xf["tot_fwd_pkts"] + Xf["tot_bwd_pkts"]).values
    sus_mask = totp >= MIN_SUS
    cache = pickle.load(open("snap_cache.pkl", "rb"))
    if PKT not in cache["snaps"]:
        raise SystemExit(f"packet {PKT} not in snap_cache")
    CACHE_COLS = cache["cols"]; M_cache, reached = cache["snaps"][PKT]
    idx65 = [CACHE_COLS.index(f) for f in COLS_65]
    name_to_65 = {f: i for i, f in enumerate(COLS_65)}
    Xfull = Xf[COLS_65].values.astype(np.float32)
    Msnap = M_cache[:, idx65].astype(np.float32)

    sh = pd.read_csv("shapes_k8_late32.csv", index_col=0)["shape_class"]
    genlate = [f for f in sh.index if sh[f] in ("genuine", "late") and f in COLS_65]

    # label encoding for xgboost
    classes = sorted(y.unique())
    y2code = {c: i for i, c in enumerate(classes)}
    enc = need_label_encoding(KIND)
    def ycode(arr): return np.array([y2code[v] for v in arr]) if enc else arr

    rows = []
    for SEED in SEEDS:
        tr, te = train_test_split(np.arange(len(Xf)), test_size=0.40,
                                  random_state=SEED, stratify=y)
        ytr = y.values[tr]
        tr_sus = tr[sus_mask[tr] & reached[tr]]
        te_sus = te[sus_mask[te] & reached[te]]
        yte = y.values[te_sus]; ytr_sus = y.values[tr_sus]

        # availability gate at packet P (sustained train)
        Xk = Msnap[tr_sus]; Xfin = Xfull[tr_sus]
        denom = np.maximum(np.abs(Xfin), 1.0)
        av = pd.Series(((np.abs(Xk - Xfin) / denom) <= DELTA).mean(axis=0), index=COLS_65)

        # per-model importance rankings
        def rank_on(Xmat, rows_idx, ylab):
            Xdf = pd.DataFrame(Xmat[rows_idx], columns=COLS_65)
            mdl = make_model(KIND, SEED).fit(Xdf, ycode(ylab))
            imp = shap_importance(mdl, Xdf, SEED).sort_values(ascending=False)
            return list(imp.index)
        rank_full = rank_on(Xfull, tr, ytr)
        rank_part = rank_on(Msnap, tr_sus, ytr_sus)

        gated_full = [f for f in rank_full if f in genlate and av[f] >= GATE]
        gated_part = [f for f in rank_part if f in genlate and av[f] >= GATE]
        gate_size = len([f for f in genlate if av[f] >= GATE])
        if gate_size < 2:
            print(f"seed {SEED}: gate<2, skip"); continue

        def fit_eval(feats, Xtr_mat, tr_rows, ytr_rows, Xte_mat):
            cidx = [name_to_65[f] for f in feats]
            Xtr = pd.DataFrame(Xtr_mat[np.ix_(tr_rows, cidx)], columns=feats)
            mdl = make_model(KIND, SEED).fit(Xtr, ycode(ytr_rows))
            pred = mdl.predict(Xte_mat[np.ix_(te_sus, cidx)])
            if enc:  # decode back for f1 against string labels
                inv = {i: c for c, i in y2code.items()}
                pred = np.array([inv[int(p)] for p in pred])
            return f1_score(yte, pred, average="macro", zero_division=0)

        for K in range(2, gate_size + 1):
            for method, pool in [("availability", gated_full), ("importance", rank_full)]:
                rows.append(dict(classifier=KIND, regime="F/F", method=method, K=K, seed=SEED,
                                 macro_f1=round(fit_eval(pool[:K], Xfull, tr, ytr, Xfull), 4)))
            for method, pool in [("availability", gated_full), ("importance", rank_full)]:
                rows.append(dict(classifier=KIND, regime="F/P", method=method, K=K, seed=SEED,
                                 macro_f1=round(fit_eval(pool[:K], Xfull, tr, ytr, Msnap), 4)))
            for method, pool in [("availability", gated_part), ("importance", rank_part)]:
                rows.append(dict(classifier=KIND, regime="P/P", method=method, K=K, seed=SEED,
                                 macro_f1=round(fit_eval(pool[:K], Msnap, tr_sus, ytr_sus, Msnap), 4)))
        print(f"seed {SEED}: gate_size={gate_size}  done")

    df = pd.DataFrame(rows)[["classifier","regime","method","K","seed","macro_f1"]]
    seedtag = "-".join(str(s) for s in SEEDS)
    out = f"/mnt/user-data/outputs/sweep_cross_{KIND}_pkt{PKT}_seeds{seedtag}.csv"
    df.to_csv(out, index=False)
    print(f"\nsaved {out}  ({len(df)} rows)")
    # quick F/P peek
    fp = df[df.regime=="F/P"].groupby(["method","K"])["macro_f1"].mean().unstack("method")
    print("F/P mean macro-F1:\n", fp.round(3).to_string())
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--classifier", required=True, choices=["random_forest","xgboost","lightgbm"])
    ap.add_argument("-p","--packet", type=int, default=8)
    ap.add_argument("-s","--seeds", type=int, nargs="+", default=[42,43,44,45,46])
    a = ap.parse_args()
    print(f"{a.classifier} | packet {a.packet} | seeds {a.seeds}")
    run(a.classifier, a.packet, a.seeds)
