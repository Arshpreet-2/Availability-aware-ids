# Data

## Included (small, per-figure CSVs)
The CSVs in this folder are the pre-computed per-seed results each figure reads, so most
figures reproduce without the large raw inputs. Note: `sweep_cat_pkt8_seeds*.csv` contains
`_MEAN` and `_SD` summary rows in the `seed` column — filter to seeds 42–46 before computing
statistics, or the SD will be wrong.

## Not included (large raw inputs)
These are omitted for size and licensing; regenerate them from the raw capture:

- `wednesday_sampled__1_.parquet` — per-flow snapshots at packets 1,2,4,8,16,32,64,128 + final values
- `shap_X.parquet`, `shap_y.parquet` — feature matrix and labels (24,221 flows × 65 features)
- `snap_cache.pkl` — cached snapshot matrices for packets 1,2,4,8,10,12,14,16,32

### How to regenerate
1. Obtain the CIC-IDS-2017 Wednesday PCAP and the corrected DistriNet/KU Leuven labels.
2. Run the checkpoint-instrumented NFStream extractor (custom plugin) to snapshot every
   feature at each packet checkpoint plus the completed-flow value.
3. Match flows to corrected labels by five-tuple + start time (nearest-time merge),
   cap BENIGN and Hulk at 5,000 each, merge '- Attempted' variants → 5 classes, 24,221 flows.
4. Drop src_port, dst_port, protocol (leaky) and the 5 constant columns → canonical 65
   (see `src/feature_set.py`, which self-validates).
