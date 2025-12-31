# make_quiz_sets.py
# Python 3.8+
# Inputs (same folder as script):
#   kanji_burden_calibrated_filtered_v2.csv
#   JmdictFurigana_For_ChatGPT.txt
#
# Outputs:
#   quiz_buckets.csv
#   bucket_summary.csv

import math
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
INPUT_CSV = "kanji_burden_calibrated_filtered_v2.csv"
FURIGANA_FILE = "JmdictFurigana_For_ChatGPT.txt"

# SeenKanji thresholds (cumulative unique kanji chars) for S0..S6
SEENKANJI_THRESHOLDS = [15, 80, 250, 620, 1000, 1500, 2000]  # S0..S6 end points

# After S6, switch to SeenComponentReading (component_string + segment_reading)
COMP_MODE = "pair"  # "pair" or "reading_only"

# Slice S7..S15 so each introduces the same amount of new comp_readings
COMP_SETS_START = 7
COMP_SETS_END = 15

# Promotion pass (top tail -> next band), processed from high to low
ENABLE_PROMOTION = True
PROMOTE_FRACTIONS_DEFAULT = 0.03
PROMOTE_FRACTIONS_BY_SETNUM = {
    # You can override specific sets here, e.g.
    # 1: 0.00,
    # 2: 0.02,
}
MIN_PROMOTE_IF_POSITIVE = 0  # set to 1 if you want "always promote at least 1 when p>0"

# Demotion scheme (fractional reference)
ENABLE_DEMOTION = True
S_START_DEMOTE = 2
S_DEMOTION_STEP = 0.6

DEMOTE_THRESHOLD = 0.90
NOISE_TYPO = 0.05  # symmetric flip noise


# -----------------------------
# Helpers: kanji + furigana parsing
# -----------------------------
SEG_RE = re.compile(r"^(?P<start>\d+)(?:-(?P<end>\d+))?:(?P<read>.*)$")


def is_kanji(ch: str) -> bool:
    o = ord(ch)
    return (
        (0x4E00 <= o <= 0x9FFF) or
        (0x3400 <= o <= 0x4DBF) or
        (0xF900 <= o <= 0xFAFF) or
        (0x20000 <= o <= 0x2A6DF) or
        (0x2A700 <= o <= 0x2B73F) or
        (0x2B740 <= o <= 0x2B81F) or
        (0x2B820 <= o <= 0x2CEAF) or
        (0x2CEB0 <= o <= 0x2EBEF)
    )


def kanji_chars(word: str) -> List[str]:
    return [ch for ch in word if is_kanji(ch)]


def load_furigana_map(path: str) -> Dict[Tuple[str, str], str]:
    m: Dict[Tuple[str, str], str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 3:
                continue
            surface = parts[0]
            reading = parts[1]
            mapping = "|".join(parts[2:])
            m[(surface, reading)] = mapping
    return m


def component_keys(surface: str, reading: str, furimap: Dict[Tuple[str, str], str]) -> List[str]:
    mapping = furimap.get((surface, reading))
    if not mapping:
        return [reading] if COMP_MODE == "reading_only" else [f"{surface}[{reading}]"]

    out: List[str] = []
    for seg in mapping.split(";"):
        if not seg:
            continue
        mo = SEG_RE.match(seg)
        if not mo:
            continue
        s = int(mo.group("start"))
        e = int(mo.group("end") or mo.group("start"))
        r = mo.group("read")
        sub = surface[s:e + 1]
        out.append(r if COMP_MODE == "reading_only" else f"{sub}[{r}]")

    if not out:
        return [reading] if COMP_MODE == "reading_only" else [f"{surface}[{reading}]"]
    return out


# -----------------------------
# Ability model (Poisson + noise)
# -----------------------------
def p_correct(N: float, b: np.ndarray) -> np.ndarray:
    P = 1.0 - np.exp(-N / b)  # success if >=1 under Poisson mean N/b
    return NOISE_TYPO + (1.0 - 2.0 * NOISE_TYPO) * P


def solve_N_for_set(burdens: np.ndarray, target: float) -> float:
    b = burdens.astype(float)
    lo = 0.0
    hi = float(np.max(b) * 50.0 + 1.0)

    for _ in range(10):
        if float(np.mean(p_correct(hi, b))) >= target:
            break
        hi *= 2.0

    for _ in range(70):
        mid = (lo + hi) / 2.0
        acc = float(np.mean(p_correct(mid, b)))
        if acc < target:
            lo = mid
        else:
            hi = mid
    return hi


def weighted_geo_mean(a: float, b: float, w: float) -> float:
    # exp((1-w)*ln(a) + w*ln(b))
    return math.exp((1.0 - w) * math.log(a) + w * math.log(b))


def N_at_fractional_level(ref: float, N_by: Dict[int, float]) -> float:
    # ref = k + frac; frac in [0,1). Use weighted geometric mean between N90(Sk) and N90(Sk+1).
    if ref <= 0.0:
        return N_by[0]
    if ref >= 15.0:
        return N_by[15]
    k = int(math.floor(ref))
    frac = ref - k
    if frac == 0.0:
        return N_by[k]
    return weighted_geo_mean(N_by[k], N_by[k + 1], frac)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    df = pd.read_csv(INPUT_CSV).sort_values("rank").reset_index(drop=True)
    furimap = load_furigana_map(FURIGANA_FILE)

    # ---- S0..S6 by SeenKanji thresholds ----
    seen_k = set()
    end_idx: List[int] = []
    t_i = 0

    for i, word in enumerate(df["word"].astype(str)):
        for ch in kanji_chars(word):
            seen_k.add(ch)

        while t_i < len(SEENKANJI_THRESHOLDS) and len(seen_k) >= SEENKANJI_THRESHOLDS[t_i]:
            end_idx.append(i)
            t_i += 1

        if t_i >= len(SEENKANJI_THRESHOLDS):
            break

    if len(end_idx) != len(SEENKANJI_THRESHOLDS):
        raise RuntimeError("Could not reach all SEENKANJI_THRESHOLDS with this dataset ordering.")

    boundaries: List[Tuple[int, int, str]] = []
    start = 0
    for s_num, end in enumerate(end_idx):
        boundaries.append((start, end, f"S{s_num}"))
        start = end + 1

    s6_end = end_idx[-1]

    # ---- S7..S15 by equal new comp_readings ----
    # Compute cum comp_readings at end of S6, and total comp_readings
    seen_c = set()
    cum6 = 0
    for i in range(0, s6_end + 1):
        w = str(df.at[i, "word"])
        r = str(df.at[i, "reading"])
        for ck in component_keys(w, r, furimap):
            if ck not in seen_c:
                seen_c.add(ck)
                cum6 += 1

    # Continue to total
    total = cum6
    for i in range(s6_end + 1, len(df)):
        w = str(df.at[i, "word"])
        r = str(df.at[i, "reading"])
        for ck in component_keys(w, r, furimap):
            if ck not in seen_c:
                seen_c.add(ck)
                total += 1

    remaining = total - cum6
    n_sets = (COMP_SETS_END - COMP_SETS_START + 1)  # 9 sets: S7..S15
    step = remaining / float(n_sets)

    # Rebuild seen_c again up to S6 for slicing pass
    seen_c.clear()
    cum = 0
    for i in range(0, s6_end + 1):
        w = str(df.at[i, "word"])
        r = str(df.at[i, "reading"])
        for ck in component_keys(w, r, furimap):
            if ck not in seen_c:
                seen_c.add(ck)
                cum += 1

    # We will create boundaries for S7..S14 using 8 targets; S15 is remainder.
    targets = [cum6 + step * k for k in range(1, n_sets)]  # k=1..8
    t_i = 0
    cur_set = COMP_SETS_START
    start = s6_end + 1

    for i in range(s6_end + 1, len(df)):
        w = str(df.at[i, "word"])
        r = str(df.at[i, "reading"])
        for ck in component_keys(w, r, furimap):
            if ck not in seen_c:
                seen_c.add(ck)
                cum += 1

        if t_i < len(targets) and cum >= targets[t_i]:
            boundaries.append((start, i, f"S{cur_set}"))
            cur_set += 1
            start = i + 1
            t_i += 1
            if cur_set == COMP_SETS_END:
                break

    # Final remainder is S15
    boundaries.append((start, len(df) - 1, f"S{COMP_SETS_END}"))

    # Assign initial buckets
    bucket = np.empty(len(df), dtype=object)
    for a, b, name in boundaries:
        bucket[a:b + 1] = name
    df["bucket"] = bucket

    # ---- Promotion pass ----
    if ENABLE_PROMOTION:
        for s in range(14, 0, -1):  # S14 -> S1
            cur = f"S{s}"
            nxt = f"S{s + 1}"
            mask = (df["bucket"] == cur)
            n = int(mask.sum())
            if n == 0:
                continue

            p = PROMOTE_FRACTIONS_BY_SETNUM.get(s, PROMOTE_FRACTIONS_DEFAULT)
            if p <= 0.0:
                continue

            k = int(math.floor(n * p))
            if k <= 0 and MIN_PROMOTE_IF_POSITIVE > 0:
                k = MIN_PROMOTE_IF_POSITIVE
            if k <= 0:
                continue

            idx = df.loc[mask, "kanji_burden_turns"].nlargest(k).index
            df.loc[idx, "bucket"] = nxt

    # ---- Compute N90 per S (pre-demotion) ----
    N_by: Dict[int, float] = {}
    for s in range(0, 16):
        b = df.loc[df["bucket"] == f"S{s}", "kanji_burden_turns"].to_numpy(dtype=float)
        N_by[s] = solve_N_for_set(b, DEMOTE_THRESHOLD) if len(b) else float("nan")

    # ---- Demotion scheme (fractional reference) ----
    if ENABLE_DEMOTION:
        for t in range(S_START_DEMOTE, 16):
            ref = (t - S_START_DEMOTE) * S_DEMOTION_STEP
            thr = N_at_fractional_level(ref, N_by)

            mask = (df["bucket"] == f"S{t}") & (df["kanji_burden_turns"].to_numpy(dtype=float) < thr)
            df.loc[mask, "bucket"] = f"L{t}"

    # Summaries
    summary_rows = []
    for name in sorted(df["bucket"].unique(), key=lambda x: (x[0], int(x[1:]))):
        sub = df[df["bucket"] == name]
        b = sub["kanji_burden_turns"].to_numpy(dtype=float)
        summary_rows.append({
            "bucket": name,
            "items": int(len(sub)),
            "burden_p50": float(np.percentile(b, 50)) if len(b) else float("nan"),
            "burden_p90": float(np.percentile(b, 90)) if len(b) else float("nan"),
            "burden_p99": float(np.percentile(b, 99)) if len(b) else float("nan"),
        })

    df.to_csv("quiz_buckets.csv", index=False)
    pd.DataFrame(summary_rows).to_csv("bucket_summary.csv", index=False)

    print("Wrote quiz_buckets.csv and bucket_summary.csv")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
