#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_quiz_sets_strong.py

Reproduces the "strong proposal" preprocessing pipeline:

Post1 (#1):
  - Ensure L-sets do not introduce new component readings in the progression stream.
  - Any item in Lk that contains a component-reading key not previously seen before Lk
    is moved to Sk (same k).
  - Component-reading keys are taken from JmdictFurigana_For_ChatGPT.txt; if missing,
    fallback is whole-word key: WORD[READING].

Post2 (#2):
  - Keep only the hardest tail of each L-set by kanji_burden_turns (quantile cutoff),
    with configurable KEEP_TOP_PERCENTAGE_L_SETS for L2..L15.
  - L2 is typically left unchanged by setting its percentage <= 0.

Outputs (in the same directory as this script):
  - moved_L_to_S_post1.csv
  - progression_of_sets_post1.txt
  - L_cull_summary_post2.csv
  - quiz_buckets_post1_post2.csv
  - expected_accuracy_table_post2.csv
  - level_thresholds_readings_and_N.csv
  - quiz_buckets_strong_Final.csv
  - progression_of_sets_post1_post2.txt
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -------------------------
# User-configurable knobs
# -------------------------

# L2, L3, ..., L15 (length must be 14)
# Interpretation:
#   - value <= 0.0  => do not cull (keep all)
#   - 0 < value < 1 => keep only the top X fraction by kanji_burden_turns (hardest tail)
KEEP_TOP_PERCENTAGE_L_SETS = [
    0,    # L2
    0.3,  # L3
    0.3,  # L4
    0.3,  # L5
    0.3,  # L6
    0.2,  # L7
    0.2,  # L8
    0.2,  # L9
    0.2,  # L10
    0.15, # L11
    0.15, # L12
    0.15, # L13
    0.15, # L14
    0.15, # L15
]

SLIP = 0.05
GUESS = 0.05


# -------------------------
# Filenames (fixed)
# -------------------------

QUIZ_BUCKETS_CSV = "quiz_buckets.csv"
PROGRESSION_TXT = "progression_of_sets.txt"
JMDICT_FURIGANA_TXT = "JmdictFurigana.txt" # not for chatgpt
WEIGHT_FILE = "weightKanjiReading_Final.csv"
WEIBULL_PARAMS_JSON = "weibull_calibration_params.json"

OUT_MOVED = "moved_L_to_S_post1.csv"
OUT_PROG_POST1 = "progression_of_sets_post1.txt"
OUT_CULL_SUMMARY = "L_cull_summary_post2.csv"
OUT_BUCKETS_POST12 = "quiz_buckets_post1_post2.csv"
OUT_EXPECTED_ACC = "expected_accuracy_table_post2.csv"
OUT_THRESHOLDS = "level_thresholds_readings_and_N.csv"
OUT_BUCKETS_FINAL = "quiz_buckets_strong_Final.csv"
OUT_PROG_POST12 = "progression_of_sets_post1_post2.txt"


# -------------------------
# Parsing JmdictFurigana
# -------------------------

def load_jmdict_furigana_map(path: Path) -> Dict[Tuple[str, str], List[str]]:
    """
    Map (surface, reading) -> list of component-reading keys like 明[めい], 白[はく].
    File format per line: surface|reading|seg;seg;...
      seg: "i:yomi" or "i-j:yomi" (inclusive indices in surface)
    """
    mp: Dict[Tuple[str, str], List[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            surface = parts[0].strip()
            reading = parts[1].strip()
            segs = parts[2].strip()
            if not surface or not reading or not segs:
                continue

            key = (surface, reading)
            if key in mp:
                # Keep the first occurrence to stay deterministic.
                continue

            comps: List[str] = []
            ok = True
            for seg in segs.split(";"):
                seg = seg.strip()
                if not seg or ":" not in seg:
                    ok = False
                    break
                idx_part, yomi = seg.split(":", 1)
                idx_part = idx_part.strip()
                yomi = yomi.strip()
                if not yomi:
                    ok = False
                    break

                if "-" in idx_part:
                    a_str, b_str = idx_part.split("-", 1)
                    try:
                        a = int(a_str)
                        b = int(b_str)
                    except ValueError:
                        ok = False
                        break
                else:
                    try:
                        a = b = int(idx_part)
                    except ValueError:
                        ok = False
                        break

                if a < 0 or b < a or b >= len(surface):
                    ok = False
                    break

                comp_surface = surface[a:b + 1]
                comps.append(f"{comp_surface}[{yomi}]")

            if ok and comps:
                mp[key] = comps

    return mp


def components_for_word(surface: str, reading: str, mp: Dict[Tuple[str, str], List[str]]) -> List[str]:
    comps = mp.get((surface, reading))
    if comps:
        return comps
    # fallback
    return [f"{surface}[{reading}]"]


# -------------------------
# Progression order
# -------------------------

def progression_order(max_level: int = 15) -> List[str]:
    order = ["S0", "S1"]
    for k in range(2, max_level + 1):
        order.append(f"L{k}")
        order.append(f"S{k}")
    return order


# -------------------------
# Post1: Move L->S if new component reading appears in L
# -------------------------

@dataclass
class MoveRecord:
    word: str
    reading: str
    from_bucket: str
    to_bucket: str
    new_component_keys: str  # ';' joined


def apply_post1_move_L_to_S(df: pd.DataFrame, comp_lists: List[List[str]]) -> Tuple[pd.DataFrame, List[MoveRecord]]:
    """
    Returns a copy of df with:
      - bucket_post1
      - moved_L_to_S_post1
    and a list of MoveRecord rows.
    """
    out = df.copy()
    out["bucket_original"] = out["bucket"]
    out["bucket_post1"] = out["bucket"]
    out["moved_L_to_S_post1"] = False

    # Store component lists as an in-memory list aligned to df.index positions
    # We'll also create a text column for easier inspection in CSV.
    comp_join = [";".join(x) for x in comp_lists]
    out["component_keys_joined"] = comp_join

    # fast index -> comp_list lookup via integer position
    # (df.index is 0..n-1 in this dataset; but handle non-default index anyway)
    idx_to_pos = {idx: i for i, idx in enumerate(out.index)}
    def comps_at_idx(idx):
        return comp_lists[idx_to_pos[idx]]

    seen_keys: set[str] = set()
    moved: List[MoveRecord] = []

    def add_bucket_components(bucket_name: str) -> None:
        mask = out["bucket_post1"].values == bucket_name
        if not mask.any():
            return
        for idx in out.index[mask]:
            for ck in comps_at_idx(idx):
                seen_keys.add(ck)

    # Process in order: S0, S1, then for each k: Lk then Sk
    add_bucket_components("S0")
    add_bucket_components("S1")

    for k in range(2, 16):
        L = f"L{k}"
        S = f"S{k}"

        # Decide moves within current L using seen_keys as of before L.
        mask_L = out["bucket_post1"].values == L
        if mask_L.any():
            idxs_L = list(out.index[mask_L])

            # Move any item that contains any component key not in seen_keys
            to_move = []
            for idx in idxs_L:
                comps = comps_at_idx(idx)
                if any(ck not in seen_keys for ck in comps):
                    to_move.append(idx)

            if to_move:
                out.loc[to_move, "bucket_post1"] = S
                out.loc[to_move, "moved_L_to_S_post1"] = True

                for idx in to_move:
                    comps = comps_at_idx(idx)
                    new_keys = [ck for ck in comps if ck not in seen_keys]
                    moved.append(MoveRecord(
                        word=str(out.at[idx, "word"]),
                        reading=str(out.at[idx, "reading"]),
                        from_bucket=L,
                        to_bucket=S,
                        new_component_keys=";".join(new_keys),
                    ))

        # Now incorporate remaining L items (post-move) into seen_keys.
        add_bucket_components(L)
        # Then incorporate Sk (including moved items) into seen_keys.
        add_bucket_components(S)

    return out, moved


# -------------------------
# Progression summary builder
# -------------------------

def compute_progression_table(df: pd.DataFrame, bucket_col: str) -> pd.DataFrame:
    """
    Builds a progression table with columns:
      set, items, cum_kanji, cum_comp_readings, rawrank_end

    cum_kanji counts unique component surfaces (left side of "[reading]").
    cum_comp_readings counts unique full component keys (surface[reading]).
    """
    seen_surfaces: set[str] = set()
    seen_keys: set[str] = set()

    rows = []
    for b in progression_order(15):
        d = df[df[bucket_col] == b]
        n_items = int(len(d))
        rawrank_end = float(d["rawrank"].max()) if n_items > 0 else float("nan")

        # Add components from this set
        for joined in d["component_keys_joined"].astype(str).tolist():
            if not joined:
                continue
            for ck in joined.split(";"):
                if not ck:
                    continue
                seen_keys.add(ck)
                surface = ck.split("[", 1)[0]
                if surface:
                    seen_surfaces.add(surface)

        rows.append({
            "set": b,
            "items": n_items,
            "cum_kanji": len(seen_surfaces),
            "cum_comp_readings": len(seen_keys),
            "rawrank_end": rawrank_end,
        })

    return pd.DataFrame(rows)


def write_progression_txt(path: Path, prog_df: pd.DataFrame) -> None:
    # Match the minimalist post1 formatting:
    # set    items  cum_kanji  cum_comp_readings  rawrank_end
    lines = []
    lines.append("set    items  cum_kanji  cum_comp_readings  rawrank_end")
    for _, r in prog_df.iterrows():
        b = str(r["set"])
        items = int(r["items"])
        ck = int(r["cum_kanji"])
        ckr = int(r["cum_comp_readings"])
        raw = r["rawrank_end"]
        raw_s = "" if (isinstance(raw, float) and math.isnan(raw)) else str(int(raw))
        lines.append(f"{b:<5} {items:>7} {ck:>9} {ckr:>18} {raw_s:>11}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------
# Post2: Cull low-burden noise from L3+
# -------------------------

@dataclass
class CullSummary:
    bucket: str
    n_total: int
    n_kept: int
    n_culled: int
    burden_cutoff: float | None


def apply_post2_cull(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[CullSummary]]:
    out = df.copy()
    out["keep_for_quiz"] = True

    if len(KEEP_TOP_PERCENTAGE_L_SETS) != 14:
        raise ValueError("KEEP_TOP_PERCENTAGE_L_SETS must have length 14 for L2..L15")

    summaries: List[CullSummary] = []

    for k in range(2, 16):
        L = f"L{k}"
        frac = float(KEEP_TOP_PERCENTAGE_L_SETS[k - 2])

        mask = out["bucket_post1"].values == L
        n_total = int(mask.sum())

        if n_total == 0:
            summaries.append(CullSummary(L, 0, 0, 0, None))
            continue

        if frac <= 0.0:
            # keep all
            summaries.append(CullSummary(L, n_total, n_total, 0, None))
            continue

        burdens = out.loc[mask, "kanji_burden_turns"].astype(float)

        # Quantile cutoff: keep items with burden >= cutoff (ties kept)
        q = 1.0 - frac
        cutoff = float(burdens.quantile(q))
        keep_mask = burdens >= cutoff

        idxs = out.loc[mask].index
        # default is True; set False where not kept
        out.loc[idxs[~keep_mask.values], "keep_for_quiz"] = False

        n_kept = int(keep_mask.sum())
        n_culled = n_total - n_kept
        summaries.append(CullSummary(L, n_total, n_kept, n_culled, cutoff))

    return out, summaries


# -------------------------
# Level thresholds (target readings -> N tokens)
# -------------------------

def load_component_probs(weight_csv: Path) -> np.ndarray:
    """
    Reads weightKanjiReading_Final.csv and returns p_i = 1/weight for each component.
    (No additional normalization; matches the library behavior.)
    """
    ps = []
    with weight_csv.open("r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        if rdr.fieldnames is None or "weight" not in rdr.fieldnames:
            raise ValueError(f"{weight_csv} must contain a 'weight' column.")
        for row in rdr:
            w_str = (row.get("weight") or "").strip()
            if not w_str:
                continue
            try:
                w = float(w_str)
            except ValueError:
                continue
            if w <= 0.0:
                continue
            p = 1.0 / w
            # clamp just in case
            if p < 0.0:
                p = 0.0
            if p > 0.999999999:
                p = 0.999999999
            ps.append(p)
    if not ps:
        raise ValueError("No probabilities loaded from weight file.")
    return np.asarray(ps, dtype=np.float64)


def expected_known_readings_vec(N: int, p: np.ndarray) -> float:
    """
    E[V(N)] = sum_i Pr(Binomial(N, p_i) >= 3)
    Vectorized for k=3.
    """
    if N < 3:
        return 0.0
    # stable t0 = (1-p)^N = exp(N*log1p(-p))
    log1mp = np.log1p(-p)
    t0 = np.exp(N * log1mp)

    inv1mp = 1.0 / (1.0 - p)
    t1 = (N * p) * t0 * inv1mp

    comb2 = (N * (N - 1)) / 2.0
    t2 = (comb2 * (p * p)) * t0 * (inv1mp * inv1mp)

    tail = 1.0 - (t0 + t1 + t2)
    # clamp numerical noise
    tail = np.clip(tail, 0.0, 1.0)
    return float(tail.sum())


def invert_unique_readings_to_N(target: float, p: np.ndarray) -> int:
    """
    Smallest integer N such that expected_known_readings_vec(N, p) >= target.
    """
    if target <= 0:
        return 0

    lo = 0
    hi = 1024  # start small and grow
    v_hi = expected_known_readings_vec(hi, p)
    while v_hi < target:
        lo = hi
        hi *= 2
        if hi > 2_000_000_000:
            raise RuntimeError("Failed to bracket target within a reasonable N range.")
        v_hi = expected_known_readings_vec(hi, p)

    # binary search
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        v_mid = expected_known_readings_vec(mid, p)
        if v_mid < target:
            lo = mid
        else:
            hi = mid
    return hi


def compute_level_targets_from_progression(prog_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract integer levels 0..15 from progression rows S0..S15.
    """
    rows = []
    for lvl in range(0, 16):
        s = f"S{lvl}"
        r = prog_df[prog_df["set"] == s]
        if r.empty:
            raise ValueError(f"Missing {s} in progression table.")
        target = int(r.iloc[0]["cum_comp_readings"])
        rows.append({"level": lvl, "target_unique_comp_readings": target})
    return pd.DataFrame(rows)


def add_half_levels(level_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with levels 0.5..15.0 step 0.5 and a column N_tokens.
    Integer levels use their own N; half levels use geometric mean.
    """
    # map integer -> N
    N_map = {int(r.level): int(r.weibull_N_tokens) for r in level_df.itertuples()}

    labels = []
    Ns = []
    for x in range(1, 31):  # 0.5..15.0 inclusive step 0.5
        lvl = x / 2.0
        if lvl.is_integer():
            N = N_map[int(lvl)]
        else:
            lo = int(math.floor(lvl))
            hi = int(math.ceil(lvl))
            N = int(round(math.sqrt(N_map[lo] * N_map[hi])))
        labels.append(lvl)
        Ns.append(N)

    return pd.DataFrame({"level": labels, "weibull_N_tokens": Ns})


# -------------------------
# Expected accuracy table
# -------------------------

def load_weibull_params(path: Path) -> Tuple[float, float]:
    js = json.loads(path.read_text(encoding="utf-8"))
    alpha = float(js["alpha"])
    beta = float(js["beta"])
    return alpha, beta


def expected_accuracy_for_set(b: np.ndarray, N: int, alpha: float, beta: float, slip: float, guess: float) -> float:
    """
    Mean p_correct across items with difficulties b, where:
      p_mastered = 1 - exp(-(beta*N/b)^alpha)
      p_correct  = guess + (1 - slip - guess) * p_mastered
    Returns percent (0..100).
    """
    # Avoid divide-by-zero (shouldn't happen)
    b = np.maximum(b, 1e-9)
    t = (beta * (N / b)) ** alpha
    p_mastered = 1.0 - np.exp(-t)
    p_correct = guess + (1.0 - slip - guess) * p_mastered
    # clamp (numerical)
    p_correct = np.clip(p_correct, 0.0, 1.0)
    return round(float(p_correct.mean() * 100.0), 2) #return float(p_correct.mean() * 100.0)


def build_expected_accuracy_table(df_quiz_pool: pd.DataFrame, level_half_df: pd.DataFrame, alpha: float, beta: float) -> pd.DataFrame:
    sets = [f"S{i}" for i in range(0, 16)] + [f"L{i}" for i in range(2, 16)]
    out = pd.DataFrame(index=sets)

    for _, row in level_half_df.iterrows():
        lvl = float(row["level"])
        N = int(row["weibull_N_tokens"])
        col = f"{lvl:.1f}"

        vals = []
        for s in sets:
            b = df_quiz_pool.loc[df_quiz_pool["bucket_post1"] == s, "kanji_burden_turns"].astype(float).to_numpy()
            if b.size == 0:
                vals.append(float("nan"))
            else:
                vals.append(expected_accuracy_for_set(b, N, alpha, beta, SLIP, GUESS))
        out[col] = vals

    out.insert(0, "set", out.index)
    out = out.reset_index(drop=True)
    return out


# -------------------------
# Main
# -------------------------

def main() -> None:
    here = Path(__file__).resolve().parent

    # Inputs
    qb_path = here / QUIZ_BUCKETS_CSV
    jf_path = here / JMDICT_FURIGANA_TXT
    w_path = here / WEIGHT_FILE
    weib_path = here / WEIBULL_PARAMS_JSON

    for p in [qb_path, jf_path, w_path, weib_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required input file: {p.name}")

    df = pd.read_csv(qb_path)

    # Load furigana mapping and compute component lists
    mp = load_jmdict_furigana_map(jf_path)
    comp_lists = [components_for_word(str(w), str(r), mp) for w, r in zip(df["word"], df["reading"])]

    # Post1: move L->S
    df1, moved = apply_post1_move_L_to_S(df, comp_lists)

    # Write moved list
    moved_df = pd.DataFrame([{
        "word": m.word,
        "reading": m.reading,
        "from_bucket": m.from_bucket,
        "to_bucket": m.to_bucket,
        "new_component_keys": m.new_component_keys,
    } for m in moved])
    moved_df.to_csv(here / OUT_MOVED, index=False, encoding="utf-8")

    # Progression post1
    prog1 = compute_progression_table(df1, bucket_col="bucket_post1")
    write_progression_txt(here / OUT_PROG_POST1, prog1)

    # Post2: cull L tails
    df2, summaries = apply_post2_cull(df1)

    # Cull summary
    summ_df = pd.DataFrame([{
        "bucket": s.bucket,
        "n_total": s.n_total,
        "n_kept": s.n_kept,
        "n_culled": s.n_culled,
        "burden_cutoff": (np.nan if s.burden_cutoff is None else float(s.burden_cutoff)),
    } for s in summaries])
    summ_df.to_csv(here / OUT_CULL_SUMMARY, index=False, encoding="utf-8")

    # Save post1+post2 buckets (do NOT delete columns)
    df2_out = df2.copy()
    # Keep bucket updated to post1 assignment for downstream convenience, but preserve originals.
    df2_out["bucket"] = df2_out["bucket_post1"]
    df2_out.to_csv(here / OUT_BUCKETS_POST12, index=False, encoding="utf-8")

    # Create final quiz pool: actually delete non-kept rows, and drop keep_for_quiz column
    df_final = df2_out[df2_out["keep_for_quiz"].astype(bool)].copy()
    if "keep_for_quiz" in df_final.columns:
        df_final = df_final.drop(columns=["keep_for_quiz"])
    df_final.to_csv(here / OUT_BUCKETS_FINAL, index=False, encoding="utf-8")

    # Progression post1+post2 (after deletions)
    # Use df_final but it no longer has keep_for_quiz; still has bucket_post1 and component_keys_joined
    prog12 = compute_progression_table(df_final, bucket_col="bucket_post1")
    write_progression_txt(here / OUT_PROG_POST12, prog12)

    # Level thresholds (integer levels)
    p = load_component_probs(w_path)
    level_targets = compute_level_targets_from_progression(prog1)

    Ns = []
    for r in level_targets.itertuples(index=False):
        N = invert_unique_readings_to_N(float(r.target_unique_comp_readings), p)
        Ns.append(N)
    level_targets["weibull_N_tokens"] = Ns
    level_targets.to_csv(here / OUT_THRESHOLDS, index=False, encoding="utf-8")

    # Half levels for expected accuracy table columns
    level_half = add_half_levels(level_targets)

    # Expected accuracy table (using final quiz pool after post2)
    alpha, beta = load_weibull_params(weib_path)
    acc_tbl = build_expected_accuracy_table(df_final, level_half, alpha, beta)
    acc_tbl.to_csv(here / OUT_EXPECTED_ACC, index=False, encoding="utf-8")

    print("Done.")
    print(f"Wrote: {OUT_MOVED}")
    print(f"Wrote: {OUT_PROG_POST1}")
    print(f"Wrote: {OUT_CULL_SUMMARY}")
    print(f"Wrote: {OUT_BUCKETS_POST12}")
    print(f"Wrote: {OUT_EXPECTED_ACC}")
    print(f"Wrote: {OUT_THRESHOLDS}")
    print(f"Wrote: {OUT_BUCKETS_FINAL}")
    print(f"Wrote: {OUT_PROG_POST12}")


if __name__ == "__main__":
    main()
