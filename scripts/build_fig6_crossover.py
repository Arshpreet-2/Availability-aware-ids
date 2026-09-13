#!/usr/bin/env python3
"""
FIGURE 6 - Genuine-6 (fixed) vs availability-gated set across decision packets 8..16.
macro-F1 averaged over the classes PRESENT at each packet (C1). Truncated at 16
(all 5 classes present through 16; GoldenEye/Slowhttptest vanish by 32).
Three shaded regions: genuine-leads / crossover / gated-pool-wins.
INPUT : figure_data/fig6_c1_data.csv   (per-seed macro_f1 by packet & method; already computed)
OUTPUT: fig6_crossover.png
To regenerate fig6_c1_data.csv from scratch you must retrain (predictions were not
otherwise saved) - see README section 'Regenerating Fig 6 data'.
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
d=pd.read_csv("figure_data/fig6_c1_data.csv")
PACKETS=[8,10,12,14,16]
mean=d.groupby(["method","packet"])["macro_f1"].mean(); sd=d.groupby(["method","packet"])["macro_f1"].std()
gs=d[d.method=="gated"].groupby("packet")["gate_size"].median()
x=np.arange(len(PACKETS))
fig,(ax,axb)=plt.subplots(2,1,figsize=(7.6,5.3),gridspec_kw={"height_ratios":[3,1]},sharex=True)
ax.axvspan(-0.3,2.0,color="#2E6B0E",alpha=0.06); ax.axvspan(2.0,3.5,color="#E0A030",alpha=0.10); ax.axvspan(3.5,4.3,color="#1f5fbf",alpha=0.07)
ax.text(0.7,0.82,"genuine-6 leads",fontsize=9,color="#2E6B0E",ha="center",style="italic",weight="bold")
ax.text(2.75,0.82,"crossover",fontsize=9,color="#B07510",ha="center",style="italic",weight="bold")
ax.text(3.9,0.30,"gated\npool wins",fontsize=9,color="#1f5fbf",ha="center",style="italic",weight="bold")
for mth,(col,ls,mk) in {"genuine6":("#2E6B0E","-","o"),"gated":("#1f5fbf","--","^")}.items():
    mm=mean[mth].reindex(PACKETS).values; ss=sd[mth].reindex(PACKETS).values
    lab="genuine-6 (fixed)" if mth=="genuine6" else "availability-gated"
    ax.plot(x,mm,linestyle=ls,marker=mk,color=col,lw=2,ms=7,markeredgecolor="white",markeredgewidth=0.7,label=lab,zorder=5)
    ax.fill_between(x,mm-ss,mm+ss,color=col,alpha=0.13,zorder=2)
ax.set_ylabel("macro-F1"); ax.grid(alpha=0.18); ax.legend(fontsize=9,loc="lower left"); ax.set_ylim(0.25,0.88); ax.set_xlim(-0.3,4.3)
axb.bar(x,gs.reindex(PACKETS).values,color="#9BB0C4",width=0.5)
axb.axhline(6,ls="--",color="#2E6B0E",lw=1)
for i in range(len(PACKETS)): axb.text(x[i],gs.reindex(PACKETS).values[i]+0.5,int(gs.reindex(PACKETS).values[i]),ha="center",fontsize=8)
axb.set_xticks(x); axb.set_xticklabels(PACKETS); axb.set_ylabel("gated set\nsize $M_k$",fontsize=9); axb.set_xlabel("decision packet")
axb.set_ylim(0,18); axb.grid(alpha=0.2,axis="y"); axb.set_xlim(-0.3,4.3)
plt.tight_layout(); plt.savefig("fig6_crossover.png",dpi=170,bbox_inches="tight"); print("saved fig6_crossover.png")
