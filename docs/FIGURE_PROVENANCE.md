# Figure Reproduction Kit — Availability-Aware Feature Selection paper

This kit lets you regenerate every figure in the paper and, crucially, tells you
**which data file each figure comes from**. Keep this with the paper. The single
most important rule is at the bottom of the provenance table: *do not compute a
figure's numbers from a different experiment's data file.*

---

## 1. Figure provenance table (READ THIS FIRST)

Each paper figure, the data file it is built from, the script that builds it, and
the key numbers it must show. If a regenerated figure does not show these numbers,
you are using the wrong data source.

| Fig | What it shows | Data file it comes from | Builder script | Key numbers to expect |
|-----|---------------|-------------------------|----------------|-----------------------|
| **1** | Four maturation categories (median availability vs packets, all flows, 25–75 percentile bands) | `inputs/snap_cache.pkl` + raw `wednesday_sampled__1_.parquet` (native packets 1–128) + `inputs/shapes_k8_late32.csv` | `figure_scripts/build_fig1_maturation.py` | genuine 6 / late 20 / trap 3 / expensive 36. Trap = syn, rst, fwd_rst_flags. Trap rises above 0.5 early then collapses. |
| **2 (left)** | Macro-F1 across 3 regimes (F/F, F/P, P/P) vs feature budget K | `figure_data/sweep_cat_pkt8_seeds42-43-44-45-46.csv` | (three-regime plot; see §4) | F/P at K=2: availability ≈ 0.56 vs importance ≈ 0.22 (gap ≈ +0.34–0.40); tie at K=6 |
| **2 (right)** | **Reproducibility: seed SD of deployed macro-F1** (availability vs importance) | `figure_data/per_seed_f1_all_packets.csv` ← **NOT the sweep file** | (per-seed SD bars; see §4) | availability SD ≈ 0.008–0.03, importance SD ≈ 0.05–0.12 at packet 8. Availability ~10× more reproducible. |
| **3** | Deployed macro-F1 vs K for three classifiers (LightGBM, RF, XGBoost) | `figure_data/robustness_cross_classifier_pkt8.csv` | `sweep_cross_classifier.py` | +0.40 (LGBM), +0.29 (RF), +0.41 (XGB) at K=2; all tie by K=6 |
| **4** | SHAP importance vs packet-8 availability scatter (per feature) | `figure_data/fig4_feature_table_leakfree.csv` | `fig45_leakfree_data.py` | most importance mass left of the 0.5 gate; W₈ = 0.94 |
| **5** | SHAP-importance rank churn across 5 seeds (left) vs fixed availability set (right) | `figure_data/churn_per_model_pkt8.csv` | `wbar_churn.py` | importance: 8–9 features rotate through top-6; availability: same 6 every seed |
| **6** | Genuine-6 (fixed) vs availability-gated set across packets 8–16 | `figure_data/fig6_c1_data.csv` | `figure_scripts/build_fig6_crossover.py` | pkt8 ≈ 0.61/0.61; pkt12 gated dips to 0.30; pkt16 crossover, gated 0.76 vs 0.68. Truncated at 16 (all 5 classes present). |

> **THE TRAP THAT CAUSED CONFUSION (documented so it never recurs):**
> Fig. 2's two panels come from **two different experiments**. The left panel
> (macro-F1 vs K) uses the K-budget sweep file. The right panel (reproducibility
> SD) uses the **per-seed fixed-6** file. If you compute the SD from the *sweep*
> file instead, availability wrongly appears *less* reproducible than importance —
> the reverse of the truth. Always take the reproducibility SD from
> `per_seed_f1_all_packets.csv` (genuine6 SD ≈ 0.008 vs importance6 SD ≈ 0.092 at
> packet 8), which matches the paper's claim.

---

## 2. What the metrics mean (so a regenerated number can be checked)

- **Availability A_k(f)** — fraction of flows in which feature *f* has reached its
  final value (within δ = 0.05, floored at 1.0) by packet *k*. Label-free.
- **W₈** — share of a complete-flow model's SHAP importance that sits on features
  with A₈ < 0.5. Paper value: 0.94 ± 0.01 (0.91–0.97 across the three classifiers).
- **macro-F1** — per-class F1 averaged equally. For Fig. 6 it is averaged over the
  classes *present at each decision packet* (all 5 through packet 16).
- **SD** — one standard deviation across the five seeds (42–46). Defined in the
  paper's Experimental Setup as "mean ± one standard deviation (SD)".
- **Sustained flows** — flows with ≥ 16 packets, the population a detector must
  actually classify early. Distinct from **all flows** (every flow, short ones
  included), used only for the maturation curve where the trap illusion appears.

---

## 3. Category classification rule (gives trap = 3)

Order matters — **check "late" before "trap"** (a feature that dips then recovers
is late, not a trap):

```
if  sustained_availability@8 >= 0.50:                      -> genuine
elif (sustained@32 - sustained@8) >= 0.20:                 -> late      # recovers
elif allflows_early_max@{1,2,4,8} >= 0.50
     and (early_max - sustained@8) >= 0.20:                -> trap      # collapses, no recovery
else:                                                      -> expensive
```

On the canonical 65 features this yields **genuine 6, late 20, trap 3
(syn, rst, fwd_rst_flags), expensive 36**. If you check "trap" before "late" you
wrongly get trap = 6 (it sweeps in fwd_pkt_len_min, pkt_len_min, fwd_seg_size_min,
which actually recover to ~0.78 by packet 32 and are therefore *late*).

---

## 4. How to regenerate each figure

Setup (once):
```
pip install pandas numpy matplotlib lightgbm xgboost shap scikit-learn pyarrow --break-system-packages
cd figure_scripts
# place the raw wednesday_sampled__1_.parquet here too (needed for Fig 1 packets 64/128)
```

**Fig 1** (self-contained, ~1 min):
```
python build_fig1_maturation.py      # -> fig1_maturation.png ; prints trap list + counts
```

**Fig 6** (uses pre-computed data, instant):
```
python build_fig6_crossover.py       # -> fig6_crossover.png
```

**Figs 2–5** use the leak-free kit scripts already in `../code/` (from Kit 2):
```
python ../code/sweep_one_packet_category.py --packet 8 --seeds 42 43 44 45 46   # feeds Fig 2 left
python ../code/sweep_cross_classifier.py --packet 8 --seeds 42 43 44 45 46      # feeds Fig 3
python ../code/fig45_leakfree_data.py                                            # feeds Fig 4
python ../code/wbar_churn.py                                                     # feeds Fig 5
```
The reproducibility panel of **Fig 2 (right)** is built from
`per_seed_f1_all_packets.csv` — the across-seed SD of the fixed genuine-6 set vs
the fixed importance-6 set at each packet. At packet 8 expect ≈ 0.008 (availability)
vs ≈ 0.092 (importance).

---

## 5. Regenerating Fig 6 data from scratch (only if needed)

`fig6_c1_data.csv` holds per-seed macro-F1 for genuine-6 vs the availability-gated
set at packets 8,10,12,14,16,32, computed over **present classes**. It required a
retrain because per-flow predictions were not otherwise saved. To regenerate, train
LightGBM per seed on the sustained population, predict on the packet-k snapshot, and
compute `f1_score(y_true, y_pred, labels=<classes present in the test slice>,
average="macro")`. The provided CSV already contains this; only regenerate if you
change the pipeline. Figures are truncated at packet 16 because GoldenEye and
Slowhttptest have ~0 flows by packet 32, which would make a 5-class macro-F1
degenerate.

---

## 6. Honest note carried from the project record

An *earlier* stability figure that once appeared in the **conference** build was
constructed by taking the real variance shape but shifting its levels to anchor at
the 0.688 headline — it was flagged at the time as "partly synthetic, regenerate
from a clean run before submission." The figures in **this** kit and in the current
paper (v6) trace to real data files as listed in §1. If you ever reuse a figure from
that older conference build, regenerate it from a clean run first.

---

## 7. File manifest

```
README.md                      <- this file
figure_scripts/
  build_fig1_maturation.py     <- Fig 1
  build_fig6_crossover.py      <- Fig 6
  feature_set.py               <- canonical 65-feature definition (shared)
figure_data/                   <- pre-computed CSVs each figure reads (see §1)
inputs/                        <- shap_X/y.parquet, snap_cache.pkl, shapes_k8_late32.csv
paper_figures/                 <- the exact PNGs used in the paper, for visual comparison
../code/                       <- Kit 2 leak-free scripts for Figs 2–5
```
