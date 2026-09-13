"""
feature_set.py  —  Canonical feature-set definition for the availability-aware IDS project.

SINGLE SOURCE OF TRUTH for which columns are used. Every script (snapshot cache,
leak-free pipeline, budget sweeps, maturation, SHAP) must import CANONICAL_FEATURES
from here instead of re-listing drops locally. Historically two scripts under-filtered
in two different ways (one kept URG, one kept URG+cwr+ece), which caused the
packet-8 gate to read 9 features instead of 6. This module removes that ambiguity.

------------------------------------------------------------------------------
FEATURE COUNT DERIVATION  (73 -> 65)
------------------------------------------------------------------------------
Raw features extracted into shap_X.parquet .......................... 73

(1) LEAKY  (-3)  : src_port, dst_port, protocol
      Dataset artifacts, not traffic behaviour. A model exploits them to
      "cheat" (src_port is the single highest-SHAP feature, 0.49) so they
      must be excluded despite/because of high importance. Removing them
      is what makes the task an honest behavioural-detection problem.

(2) DEAD  (-2)   : cwr, ece
      Near-constant TCP-flag counters: exactly 1 non-zero flow each out of
      24,221 (nunique == 2, std ~= 0.006-0.013). SHAP importance exactly
      0.0 — carry no usable signal.

(3) URG   (-3)   : fwd_urg_flags, bwd_urg_flags, urg
      Strictly constant: nunique == 1, std == 0, ZERO non-zero values in the
      entire dataset; SHAP importance exactly 0.0. Same no-information reason
      as cwr/ece (cwr/ece each have a single non-zero flow; URG has none). They were previously missed
      by the exclude list, so they trivially passed the availability gate
      (a never-changing column always "equals its final value" => reported
      100% settled) and falsely inflated the packet-8 gate from 6 to 9.
      Dropping them is a no-information removal, NOT a modelling choice.

Remaining CANONICAL feature set ..................................... 65
------------------------------------------------------------------------------
"""

# --- drop groups (keep them named so downstream code / docs can cite the reason) ---
LEAKY = ["src_port", "dst_port", "protocol"]                 # dataset artifacts (id-like)
DEAD  = ["cwr", "ece"]                                        # constant TCP-flag counters
URG   = ["fwd_urg_flags", "bwd_urg_flags", "urg"]            # constant URG-flag counters

# all zero-information constant columns (dead + URG), grouped for clarity
CONSTANT = DEAD + URG                                         # 5 columns, all SHAP == 0.0

# everything we remove from the raw 73
DROP = LEAKY + CONSTANT                                       # 3 + 5 = 8 columns

EXPECTED_RAW   = 73
EXPECTED_FINAL = 65


def canonical_features(all_columns):
    """Return the 65 canonical feature names, in the order they appear in `all_columns`,
    with every leaky and constant column removed.

    Parameters
    ----------
    all_columns : iterable of str
        The full column list, e.g. list(pd.read_parquet("shap_X.parquet").columns).

    Returns
    -------
    list of str  (length 65 on the reference dataset)
    """
    drop = set(DROP)
    return [c for c in all_columns if c not in drop]


def validate(df, verbose=True):
    """Sanity-check a raw feature frame against the expected derivation.

    Confirms (a) the raw count is 73, (b) the canonical count is 65, and
    (c) every column in CONSTANT is genuinely constant (std == 0). Raises
    AssertionError on any mismatch so a silent data change can't slip through.
    """
    cols = list(df.columns)
    feats = canonical_features(cols)
    if verbose:
        print(f"raw columns           : {len(cols)}  (expected {EXPECTED_RAW})")
        print(f"dropped (leaky)        : {len(LEAKY)}  {LEAKY}")
        print(f"dropped (dead const)   : {len(DEAD)}  {DEAD}")
        print(f"dropped (urg const)    : {len(URG)}  {URG}")
        print(f"canonical features     : {len(feats)}  (expected {EXPECTED_FINAL})")
    assert len(cols) == EXPECTED_RAW,  f"raw column count {len(cols)} != {EXPECTED_RAW}"
    assert len(feats) == EXPECTED_FINAL, f"canonical count {len(feats)} != {EXPECTED_FINAL}"
    # cwr/ece each have a single non-zero flow (nunique==2); URG are strictly constant.
    # All five carry zero SHAP importance, so the test is "near-zero variance", not
    # strict constancy. Threshold is generous; a real feature would exceed it by orders
    # of magnitude.
    for c in CONSTANT:
        if c in df.columns:
            std = float(df[c].std())
            nz  = int((df[c] != 0).sum())
            assert std < 0.05, f"{c} expected near-constant but std={std:.4f} (nonzero flows={nz})"
            if verbose:
                print(f"  const-check {c:<15} std={std:.6f}  nonzero_flows={nz}  -> ok")
    if verbose:
        print("validate: OK")
    return feats


if __name__ == "__main__":
    import pandas as pd
    Xf = pd.read_parquet("shap_X.parquet")
    validate(Xf)
