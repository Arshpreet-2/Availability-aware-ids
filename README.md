# Availability-Aware Feature Selection for Early Flow-Based DoS Detection

Code and figure-reproduction materials for the conference paper *"Availability-Aware
Feature Selection for Early Flow-Based DoS Detection."*

The paper reframes feature **cost** as feature **availability** — how many packets a
feature needs before its value can be trusted — and uses it as a label-free gate ahead
of importance ranking for early DoS detection on CIC-IDS-2017.

## What's here

```
src/                 feature_set.py — canonical 65-feature definition (self-validating)
scripts/             one script per figure / experiment (see table below)
data/                pre-computed per-seed CSVs each figure reads
figures/             the exact figures used in the paper (for comparison)
docs/                FIGURE_PROVENANCE.md — which data file each figure comes from
requirements.txt     Python dependencies
```

## Figure → script → data map

| Figure | Script | Reads |
|--------|--------|-------|
| Fig 1 — maturation categories | `scripts/build_fig1_maturation.py` | raw parquet + `snap_cache.pkl` + `shapes_k8_late32.csv` |
| Fig 2 — 3 regimes + reproducibility | `scripts/sweep_one_packet_category.py` | `data/sweep_cat_pkt8_seeds*.csv` |
| Fig 3 — three classifiers | `scripts/sweep_cross_classifier.py` | `data/robustness_cross_classifier_pkt8.csv` |
| Figs 4–5 — scatter, churn | `scripts/fig45_leakfree_data.py`, `scripts/wbar_churn.py` | `data/fig4_feature_table_leakfree.csv`, `data/churn_per_model_pkt8.csv` |
| Fig 6 — crossover | `scripts/build_fig6_crossover.py` | `data/fig6_c1_data.csv` |

Full provenance, key numbers to expect, and gotchas are in **`docs/FIGURE_PROVENANCE.md`** —
read it before regenerating anything. In particular, `sweep_cat_pkt8_seeds*.csv` contains
`_MEAN` / `_SD` summary rows that must be filtered out (keep seeds 42–46 only) before
computing statistics.

## Reproduce

```bash
pip install -r requirements.txt
# regenerate a figure that uses pre-computed data (fast):
python scripts/build_fig6_crossover.py
# regenerate from scratch (retrains models, slower):
python scripts/sweep_one_packet_category.py --packet 8 --seeds 42 43 44 45 46
```

## Data

The large raw inputs (`wednesday_sampled__1_.parquet`, `shap_X.parquet`,
`shap_y.parquet`, `snap_cache.pkl`) are **not** committed here because of size. See
`data/DATA.md` for how to obtain or regenerate them from the raw CIC-IDS-2017 Wednesday
capture with the corrected DistriNet labels. The small per-figure CSVs in `data/` are
included so most figures reproduce without them.

## Method in one paragraph

A feature's **availability** A_k(f) is the fraction of flows in which it has reached its
final value by packet k, measured over the sustained-flow population (flows ≥16 packets).
Selection keeps the top-K most important features **among those with A_8 ≥ 0.5**, so the
gate is label-free and the ranking uses labels. On CIC-IDS-2017 Wednesday, ~94% of a
complete-flow model's SHAP importance sits on features that have not settled by packet 8;
availability-gated selection matches importance-based selection in deployed accuracy while
being ~10× more reproducible across seeds.

## Citation

Add your BibTeX here once the paper is published.

## License

See `LICENSE`.
