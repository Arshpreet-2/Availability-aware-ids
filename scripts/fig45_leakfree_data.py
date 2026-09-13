#!/usr/bin/env python3
"""
fig45_leakfree_data.py
Regenerate the per-feature and per-category data for Fig. 4 and Fig. 5 from the
LEAK-FREE pipeline, so both trace to the same run as Table I and Figs 1-3.

Fig. 4 needs, per feature: packet-8 availability, LEAK-FREE mean SHAP importance
       (per-seed on train split, averaged), and maturation category.
Fig. 5 needs, per category: mean availability at each packet checkpoint.

Leak-free importance = mean over seeds of mean|SHAP| computed on that seed's TRAIN
split only (never a whole-data ranking). Availability = fraction of sustained flows
where the feature is within DELTA of its final value at the checkpoint (identical to
the rest of the pipeline). Categories come from shapes_k8_late32.csv but intersected
with the canonical-65 (leaky/constant excluded).

Outputs:
    fig4_feature_table_leakfree.csv   feature, availability_p8, importance_leakfree, category
    fig5_category_curves_leakfree.csv category, packet, mean_availability
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd, shap
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from feature_set import canonical_features

SEEDS=[42,43,44,45,46]; MIN_SUS=16; DELTA=0.05; SHAP_BG=3000
PACKETS=[1,2,4,8,10,12,14,16,32]

def merge(s): return s.str.replace(" - Attempted","",regex=False)
def lgbm(s): return lgb.LGBMClassifier(n_estimators=250,learning_rate=0.1,num_leaves=31,
                                       class_weight="balanced",random_state=s,n_jobs=-1,verbose=-1)

Xf=pd.read_parquet("shap_X.parquet").reset_index(drop=True)
y=merge(pd.read_parquet("shap_y.parquet")["Label"]).reset_index(drop=True)
COLS=canonical_features(Xf.columns)
totp=(Xf["tot_fwd_pkts"]+Xf["tot_bwd_pkts"]).values; sus=totp>=MIN_SUS
cache=pickle.load(open("snap_cache.pkl","rb")); CC=cache["cols"]
idx={f:CC.index(f) for f in COLS}
Xfull=Xf[COLS].values.astype(np.float32)

# ---- leak-free per-seed importance (mean|SHAP| on train split), averaged ----
imp_seeds=[]
for seed in SEEDS:
    tr,te=train_test_split(np.arange(len(Xf)),test_size=0.40,random_state=seed,stratify=y)
    Xdf=Xf.iloc[tr][COLS]
    m=lgbm(seed).fit(Xdf,y.iloc[tr])
    bg=Xdf.sample(n=min(SHAP_BG,len(tr)),random_state=seed)
    sv=shap.TreeExplainer(m).shap_values(bg)
    if isinstance(sv,list): impv=np.mean([np.abs(a).mean(0) for a in sv],0)
    elif np.abs(sv).ndim==3: impv=np.abs(sv).mean((0,2))
    else: impv=np.abs(sv).mean(0)
    imp_seeds.append(pd.Series(impv,index=COLS))
    print(f"seed {seed}: importance computed")
imp_leakfree=pd.concat(imp_seeds,axis=1).mean(axis=1)   # mean across seeds

# ---- packet-8 availability on sustained flows (whole sustained pop) ----
def avail_at(pkt):
    M,reached=cache["snaps"][pkt]
    mask=sus & reached
    Msnap=M[:,[idx[f] for f in COLS]].astype(np.float32)
    Xk=Msnap[mask]; Xfin=Xfull[mask]; den=np.maximum(np.abs(Xfin),1.0)
    return pd.Series(((np.abs(Xk-Xfin)/den)<=DELTA).mean(0),index=COLS)
av8=avail_at(8)

# ---- categories (shapes intersected with canonical-65) ----
sh=pd.read_csv("shapes_k8_late32.csv",index_col=0)["shape_class"]
cat=pd.Series({f:(sh[f] if f in sh.index else "n/a") for f in COLS})

# ---- Fig 4 table ----
fig4=pd.DataFrame({"feature":COLS,
                   "availability_p8":av8.reindex(COLS).round(4).values,
                   "importance_leakfree":imp_leakfree.reindex(COLS).round(4).values,
                   "category":cat.reindex(COLS).values}).sort_values("importance_leakfree",ascending=False)
fig4.to_csv("/mnt/user-data/outputs/fig4_feature_table_leakfree.csv",index=False)

# W-bar leak-free over the genuine gate: mass on features with availability<0.5
Wbar = imp_leakfree[av8<0.5].sum()/imp_leakfree.sum()

# ---- Fig 5 category curves: mean availability per category per packet ----
rows=[]
for pkt in PACKETS:
    a=avail_at(pkt)
    for c in ["genuine","late","trap","expensive"]:
        feats=[f for f in COLS if cat[f]==c]
        if feats: rows.append(dict(category=c,packet=pkt,mean_availability=round(a[feats].mean(),4),n=len(feats)))
fig5=pd.DataFrame(rows)
fig5.to_csv("/mnt/user-data/outputs/fig5_category_curves_leakfree.csv",index=False)

print("\n=== Fig 4 leak-free TOP 10 by importance ===")
print(fig4.head(10).to_string(index=False))
print(f"\nleak-free W-bar (importance mass on A_8<0.5, all 65) = {Wbar:.3f}")
print("(paper quotes 0.76; compare)")
print("\n=== Fig 5 category means at packets 1/8/32 ===")
print(fig5[fig5.packet.isin([1,8,32])].pivot(index="category",columns="packet",values="mean_availability").to_string())
print("\nsaved fig4_feature_table_leakfree.csv, fig5_category_curves_leakfree.csv")
