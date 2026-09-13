#!/usr/bin/env python3
"""
sweep_one_packet_category.py
============================================================================
ONE packet, ONE OR MORE seeds. Leak-free budget sweep of the AVAILABILITY-AWARE
(category) selection vs the IMPORTANCE baseline, SUSTAINED population, canonical
65-feature set, across THREE train/test regimes.

THE THREE REGIMES
-----------------
  F/F  train FULL (final-value) features   -> test FULL  features   (upper bound)
  F/P  train FULL features                 -> test at packet P       (deployment gap)
  P/P  train at packet P (partial values)  -> test at packet P       (deploy-realistic)

PER SEED, PER REGIME
--------------------
  1. Leak-free split; SHAP importance computed on TRAIN split only.
     F/F & F/P rank on FULL train values; P/P ranks on packet-P train values.
  2. CATEGORY POOL = (genuine+late in shapes_k8_late32.csv) INTERSECT canonical-65.
  3. AVAILABILITY GATE at packet P, SUSTAINED (>=16 pkts): keep category features
     with A_P(f) >= 0.5. Prints how many are eligible.
  4. Sweep K = 2..gate_size:
       availability : top-K of gated pool, ordered by IMPORTANCE
       importance   : top-K by importance over ALL 65 features
  5. Record 6 metrics per (regime, method, K, seed).

Aggregated to mean +/- SD across the seeds you pass. CSV keeps per-seed rows
plus _MEAN and _SD rows.

USAGE
-----
    python sweep_one_packet_category.py --packet 8 --seeds 42
    python sweep_one_packet_category.py --packet 8 --seeds 42 43 44 45 46
    python sweep_one_packet_category.py -p 16 -s 42 43 44

Outputs (names encode packet + seed list):
    sweep_cat_pkt<P>_seeds<s1-s2-...>.csv
    sweep_cat_pkt<P>_seeds<s1-s2-...>.png

INPUTS (same dir): feature_set.py, shap_X.parquet, shap_y.parquet,
                   snap_cache.pkl, shapes_k8_late32.csv
============================================================================
"""
import warnings; warnings.filterwarnings("ignore")
import argparse, pickle
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb, shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, accuracy_score, balanced_accuracy_score,
                             precision_score, recall_score)
from feature_set import canonical_features

MIN_SUS = 16
DELTA   = 0.05
GATE    = 0.50
SHAP_BG = 3000
REGIMES = ["F/F", "F/P", "P/P"]
METRIC_COLS = ["macro_f1","weighted_f1","accuracy","balanced_acc","macro_prec","macro_rec"]

def lgbm(seed):
    return lgb.LGBMClassifier(n_estimators=250, learning_rate=0.1, num_leaves=31,
                              class_weight="balanced", random_state=seed,
                              n_jobs=-1, verbose=-1)

def merge_attempted(s):
    return s.str.replace(" - Attempted", "", regex=False)

def metrics(ytrue, pred):
    return dict(
        macro_f1     = f1_score(ytrue, pred, average="macro", zero_division=0),
        weighted_f1  = f1_score(ytrue, pred, average="weighted", zero_division=0),
        accuracy     = accuracy_score(ytrue, pred),
        balanced_acc = balanced_accuracy_score(ytrue, pred),
        macro_prec   = precision_score(ytrue, pred, average="macro", zero_division=0),
        macro_rec    = recall_score(ytrue, pred, average="macro", zero_division=0),
    )

def run_seed(PKT, SEED, Xf, y, COLS_65, sus_mask, cache, genlate):
    CACHE_COLS = cache["cols"]
    M_cache, reached = cache["snaps"][PKT]
    idx65 = [CACHE_COLS.index(f) for f in COLS_65]
    name_to_65 = {f: i for i, f in enumerate(COLS_65)}
    Xfull = Xf[COLS_65].values.astype(np.float32)     # full/final values
    Msnap = M_cache[:, idx65].astype(np.float32)      # packet-P values

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

    def rank_from(Xmat, rows, ycol):
        Xdf = pd.DataFrame(Xmat[rows], columns=COLS_65)
        m = lgbm(SEED).fit(Xdf, ycol)
        bg = Xdf.sample(n=min(SHAP_BG, len(rows)), random_state=SEED)
        sv = shap.TreeExplainer(m).shap_values(bg)
        if isinstance(sv, list):   impv = np.mean([np.abs(a).mean(0) for a in sv], 0)
        elif np.abs(sv).ndim == 3: impv = np.abs(sv).mean((0, 2))
        else:                      impv = np.abs(sv).mean(0)
        return pd.Series(impv, index=COLS_65).sort_values(ascending=False)

    imp_full = rank_from(Xfull, tr, ytr)          # for F/F, F/P
    imp_part = rank_from(Msnap, tr_sus, ytr_sus)  # for P/P

    def gated_pool(imp_series):
        rank = list(imp_series.index)
        return [f for f in rank if f in genlate and av[f] >= GATE], rank
    gated_full, rank_full = gated_pool(imp_full)
    gated_part, rank_part = gated_pool(imp_part)
    gate_size = len([f for f in genlate if av[f] >= GATE])

    rows = []
    if gate_size < 2:
        return rows, gate_size, len(genlate), av, gated_full

    def fit_eval(feats, Xtr_mat, tr_rows, ytr_rows, Xte_mat):
        cidx = [name_to_65[f] for f in feats]
        Xtr = pd.DataFrame(Xtr_mat[np.ix_(tr_rows, cidx)], columns=feats)
        mm = lgbm(SEED).fit(Xtr, ytr_rows)
        pred = mm.predict(Xte_mat[np.ix_(te_sus, cidx)])
        return metrics(yte, pred)

    for K in range(2, gate_size + 1):
        for method, pool in [("availability", gated_full), ("importance", rank_full)]:
            rows.append(dict(regime="F/F", method=method, K=K, seed=SEED,
                             **fit_eval(pool[:K], Xfull, tr, ytr, Xfull)))
        for method, pool in [("availability", gated_full), ("importance", rank_full)]:
            rows.append(dict(regime="F/P", method=method, K=K, seed=SEED,
                             **fit_eval(pool[:K], Xfull, tr, ytr, Msnap)))
        for method, pool in [("availability", gated_part), ("importance", rank_part)]:
            rows.append(dict(regime="P/P", method=method, K=K, seed=SEED,
                             **fit_eval(pool[:K], Msnap, tr_sus, ytr_sus, Msnap)))
    return rows, gate_size, len(genlate), av, gated_full

def run(PKT, SEEDS):
    Xf = pd.read_parquet("shap_X.parquet").reset_index(drop=True)
    y  = merge_attempted(pd.read_parquet("shap_y.parquet")["Label"]).reset_index(drop=True)
    COLS_65 = canonical_features(Xf.columns)
    totp = (Xf["tot_fwd_pkts"] + Xf["tot_bwd_pkts"]).values
    sus_mask = totp >= MIN_SUS
    cache = pickle.load(open("snap_cache.pkl", "rb"))
    if PKT not in cache["snaps"]:
        raise SystemExit(f"packet {PKT} not in snap_cache (have {sorted(cache['snaps'])})")
    sh = pd.read_csv("shapes_k8_late32.csv", index_col=0)["shape_class"]
    genlate = [f for f in sh.index if sh[f] in ("genuine","late") and f in COLS_65]

    all_rows=[]; gate_size=None; av_ref=None; gated_ref=None; n_pool=len(genlate)
    for SEED in SEEDS:
        rr, gs, np_, av, gated = run_seed(PKT, SEED, Xf, y, COLS_65, sus_mask, cache, genlate)
        all_rows += rr; gate_size=gs; av_ref=av; gated_ref=gated
        print(f"seed {SEED}: availability-eligible at packet {PKT} = {gs} of {np_}")

    print(f"\n{'='*66}")
    print(f"packet {PKT} | seeds {SEEDS} | sustained (>= {MIN_SUS} pkts)")
    print(f"category pool (genuine+late \u2229 65) : {n_pool}")
    print(f"AVAILABILITY-ELIGIBLE (A_{PKT} >= {GATE:.2f}) : {gate_size}")
    if gated_ref:
        print("eligible features (importance order):")
        for f in gated_ref: print(f"    {f:<22} A_{PKT}={av_ref[f]:.3f}")
    print('='*66)
    if not all_rows: raise SystemExit("gate too small to sweep.")

    df = pd.DataFrame(all_rows)
    g = df.groupby(["regime","method","K"])[METRIC_COLS]
    mean = g.mean().reset_index(); sd = g.std().reset_index().fillna(0.0)
    mean["seed"]="_MEAN"; sd["seed"]="_SD"
    out = pd.concat([df, mean[df.columns], sd[df.columns]], ignore_index=True)
    for c in METRIC_COLS: out[c]=out[c].round(4)

    seedtag="-".join(str(s) for s in SEEDS)
    csv_name=f"/mnt/user-data/outputs/sweep_cat_pkt{PKT}_seeds{seedtag}.csv"
    out.to_csv(csv_name,index=False)
    print(f"\nsaved CSV: {csv_name}  ({len(df)} per-seed rows + mean/SD)")

    # CSV keeps all 6 metrics; the FIGURE shows only these 3 (as requested)
    metric_titles=[("macro_f1","macro-F1"),("balanced_acc","balanced accuracy"),
                   ("macro_prec","macro precision")]
    GREEN,RED="#2E6B0E","#C0392B"
    reg_ls={"F/F":"-","F/P":"--","P/P":":"}; reg_mk={"F/F":"o","F/P":"s","P/P":"^"}
    fig,axes=plt.subplots(1,3,figsize=(15,5.2),sharex=True)
    Ks=sorted(df["K"].unique())
    for ax,(col,title) in zip(axes.ravel(),metric_titles):
        for reg in REGIMES:
            for method,color in [("availability",GREEN),("importance",RED)]:
                m=mean[(mean.regime==reg)&(mean.method==method)].sort_values("K")
                s=sd[(sd.regime==reg)&(sd.method==method)].sort_values("K")
                if m.empty: continue
                ax.plot(m["K"],m[col],reg_ls[reg]+reg_mk[reg],color=color,lw=1.8,ms=4,
                        label=f"{method[:4]} {reg}")
                if len(SEEDS)>1:
                    ax.fill_between(m["K"],m[col]-s[col].values,m[col]+s[col].values,color=color,alpha=0.08)
        ax.set_title(title,fontsize=11); ax.set_ylim(0,1); ax.grid(alpha=0.3); ax.set_xticks(Ks)
    for ax in axes: ax.set_xlabel("feature budget K",fontsize=10)
    axes[0].legend(fontsize=7,ncol=2,loc="lower right")
    fig.suptitle(f"Availability-aware (green) vs importance-only (red) | packet {PKT} | "
                 f"F/F solid, F/P dashed, P/P dotted\n"
                 f"sustained | {len(SEEDS)} seed(s): {SEEDS} | "
                 f"availability-eligible at packet {PKT} = {gate_size} of {n_pool}",fontsize=11)
    fig.tight_layout(rect=[0,0,1,0.93])
    png_name=f"/mnt/user-data/outputs/sweep_cat_pkt{PKT}_seeds{seedtag}.png"
    fig.savefig(png_name,dpi=150,bbox_inches="tight"); plt.close()
    print(f"saved figure: {png_name}")

    print("\nmacro-F1 (mean over seeds):")
    for reg in REGIMES:
        piv=mean[mean.regime==reg].pivot(index="K",columns="method",values="macro_f1")
        print(f"  [{reg}]  "+"  ".join(f"K{K}:av={piv.loc[K,'availability']:.3f}/im={piv.loc[K,'importance']:.3f}"
                                        for K in piv.index))
    return csv_name,png_name

if __name__=="__main__":
    ap=argparse.ArgumentParser(description="One-packet, multi-seed, 3-regime category-vs-importance budget sweep.")
    ap.add_argument("-p","--packet",type=int,required=True,help="decision packet (e.g. 8,16,32)")
    ap.add_argument("-s","--seeds",type=int,nargs="+",required=True,
                    help="one or more seeds, e.g. --seeds 42 43 44 45 46")
    args=ap.parse_args()
    print(f"running packet {args.packet} over {len(args.seeds)} seed(s): {args.seeds}")
    run(args.packet,args.seeds)
