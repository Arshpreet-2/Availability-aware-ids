#!/usr/bin/env python3
"""
FIGURE 1 - Four maturation categories (median availability vs packets, all flows).
Print-safe: distinct colour + line style + marker per category. 25th-75th percentile bands.
Categories via Rule B (late-before-trap): trap = syn, rst, fwd_rst_flags (3).
INPUT : inputs/shap_X.parquet, inputs/snap_cache.pkl, inputs/shapes_k8_late32.csv,
        ../inputs/wednesday_sampled__1_.parquet (raw, for packets 64/128)
OUTPUT: fig1_maturation.png
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, numpy as np, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from feature_set import canonical_features
DELTA=0.05; MIN_SUS=16; PACKETS=[1,2,4,8,16,32,64,128]
RAW="wednesday_sampled__1_.parquet"   # place raw parquet alongside, has native packets 1..128
w=pd.read_parquet(RAW)
finals=[]; snaps={p:[] for p in PACKETS}; COLS_all=None
for i in range(len(w)):
    fin=json.loads(w["final"].iloc[i]); sn=json.loads(w["snaps"].iloc[i]); finals.append(fin)
    if COLS_all is None: COLS_all=list(fin.keys())
    for p in PACKETS: snaps[p].append(sn.get(str(p),{}))
finals=pd.DataFrame(finals)
COLS=[c for c in canonical_features(pd.Index(COLS_all)) if c in finals.columns]
totp=w["final_np"].values; sus=totp>=MIN_SUS; allm=np.ones(len(w),bool)
Fin=finals[COLS].astype(float).values; den=np.maximum(np.abs(Fin),1.0)
def av(j,mask,p):
    col=np.array([snaps[p][i].get(COLS[j],np.nan) for i in range(len(w))])
    fv=Fin[:,j]; dn=den[:,j]; ok=~np.isnan(col); m=mask&ok
    return (np.abs(col[m]-fv[m])/dn[m]<=DELTA).mean() if m.sum() else np.nan
# classify (Rule B: late before trap)
AV=0.5; DROP=0.2; RISE=0.2; cats={}
for j,f in enumerate(COLS):
    sus8=av(j,sus,8); sus32=av(j,sus,32); early=max(av(j,allm,p) for p in [1,2,4,8])
    if sus8>=AV: cats[f]="genuine"
    elif (sus32-sus8)>=RISE: cats[f]="late"
    elif early>=AV and (early-sus8)>=DROP: cats[f]="trap"
    else: cats[f]="expensive"
cat_list=["genuine","late","trap","expensive"]
pct={c:{"lo":[],"med":[],"hi":[]} for c in cat_list}
def av_all(j,p):
    col=np.array([snaps[p][i].get(COLS[j],np.nan) for i in range(len(w))])
    fv=Fin[:,j]; dn=den[:,j]; ok=~np.isnan(col)
    return (np.abs(col[ok]-fv[ok])/dn[ok]<=DELTA).mean() if ok.sum() else np.nan
for p in PACKETS:
    pf={c:[] for c in cat_list}
    for j,f in enumerate(COLS): pf[cats[f]].append(av_all(j,p))
    for c in cat_list:
        v=np.array([x for x in pf[c] if np.isfinite(x)])
        pct[c]["lo"].append(np.percentile(v,25)); pct[c]["med"].append(np.percentile(v,50)); pct[c]["hi"].append(np.percentile(v,75))
style={"genuine":("#2E6B0E","-","o"),"late":("#E08214","--","s"),"trap":("#C0392B","-.","^"),"expensive":("#6A51A3",":","D")}
nc={c:sum(1 for f in COLS if cats[f]==c) for c in cat_list}
fig,ax=plt.subplots(figsize=(7.2,4.6))
for c in cat_list:
    col,ls,mk=style[c]; med=np.array(pct[c]["med"])
    ax.plot(PACKETS,med,linestyle=ls,marker=mk,color=col,lw=2,ms=6,markeredgecolor="white",markeredgewidth=0.6,label=f"{c} (n={nc[c]})")
    ax.fill_between(PACKETS,pct[c]["lo"],pct[c]["hi"],color=col,alpha=0.13)
ax.axhline(0.5,ls=(0,(2,2)),color="grey",lw=1); ax.text(1.05,0.52,"0.5",fontsize=8,color="grey")
ax.set_xscale("log",base=2); ax.set_xticks(PACKETS); ax.set_xticklabels(PACKETS)
ax.set_xlabel("packets seen  (log scale)"); ax.set_ylabel("availability  $A_k(f)$")
ax.set_ylim(0,1.02); ax.legend(fontsize=9,loc="center right"); ax.grid(alpha=0.22)
plt.tight_layout(); plt.savefig("fig1_maturation.png",dpi=170,bbox_inches="tight")
print("trap =",[f for f in COLS if cats[f]=="trap"]); print("counts",nc)
