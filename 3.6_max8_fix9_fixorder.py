# compute_kanji_burden_3.6.py
# Parallel (6 cores) exact DP for m<=8; for m>=9 use heuristic (heur_est_rms_turns) as kanji_burden_turns.
#
# Inputs (expected in same folder as this script):
#   - kanjiwords_jmdictslice.csv   (columns: word,reading,freq,rawrank)
#   - JmdictFurigana.txt           (lines: surface|reading|seg;seg;... where seg is i:j or i-k:j)
#   - _weightKanjireading.csv      (columns include a key like "component" or similar; see POSSIBLE_KEY_COLS)
#
# Output:
#   - kanji_burden.csv with columns:
#       word,reading,n_components,rank,kanji_rank,rawrank,kanji_burden_turns
#
# Notes:
# - rank is "1224 competition ranking" on rawrank (ascending).
# - kanji_rank is "1224 competition ranking" on kanji_burden_turns (ascending).
# - kanji_burden_turns is rounded to DECIMAL_BURDEN places for output AND ranking.
#
# Yuusei: per your request, NO fallback sample filenames are used.

from __future__ import annotations

import csv
import math
import os
import multiprocessing as mp
from typing import Dict, List, Tuple


# -----------------------
# User-configurable knobs
# -----------------------
THRESHOLD_LEARN = 3           # k
CPU_CORES = 6
CHUNKSIZE = 50                # lower to reduce tail latency; 10-100 is reasonable
MAX_M_EXACT = 8               # compute exact DP only when m <= this; else heuristic
DECIMAL_BURDEN = 3            # rounding of kanji_burden_turns

OUT_MAIN = "kanji_burden.csv"

KANJIWORDS_CSV = "kanjiwords_jmdictslice.csv"
FURIGANA_TXT = "JmdictFurigana.txt"
WEIGHT_READINGS_CSV = "weightKanjireading_Final.csv"#"_weightKanjireading.csv"

# Which column in _weightKanjireading.csv contains the component key (like 大人[おとな])?
# Try a few common names; if none are found, fall back to first column.
POSSIBLE_KEY_COLS = ("component", "key", "reading_key", "kanji_reading", "form", "pair", "kanjireading")


# -----------------------
# Globals for worker processes (read-only)
# -----------------------
G_INVERTED_INDEX: List[List[int]] = []
G_P_WORD: List[float] = []
G_K: int = 3


def _init_worker(inverted_index: List[List[int]], p_word: List[float], k: int) -> None:
    global G_INVERTED_INDEX, G_P_WORD, G_K
    G_INVERTED_INDEX = inverted_index
    G_P_WORD = p_word
    G_K = k


# -----------------------
# File helpers
# -----------------------
def require_file(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required input file: '{path}' (expected in same folder as script)")
    return path


# -----------------------
# Furigana parsing -> component keys
# -----------------------
def parse_furigana_line(line: str) -> Tuple[str, str, List[Tuple[int, int, str]]]:
    """
    Returns (surface, reading, segments)
    segments = list of (start_idx, end_idx, kana_chunk)
    """
    line = line.strip()
    if not line:
        return ("", "", [])
    parts = line.split("|")
    if len(parts) < 2:
        return ("", "", [])
    surface = parts[0]
    reading = parts[1]
    segs_raw = parts[2] if len(parts) >= 3 else ""
    segs: List[Tuple[int, int, str]] = []

    if segs_raw:
        for seg in segs_raw.split(";"):
            seg = seg.strip()
            if not seg or ":" not in seg:
                continue
            idx_part, kana = seg.split(":", 1)
            idx_part = idx_part.strip()
            kana = kana.strip()
            if not kana:
                continue

            if "-" in idx_part:
                a_str, b_str = idx_part.split("-", 1)
                try:
                    a = int(a_str)
                    b = int(b_str)
                except ValueError:
                    continue
            else:
                try:
                    a = int(idx_part)
                    b = a
                except ValueError:
                    continue

            segs.append((a, b, kana))
    return (surface, reading, segs)


def segments_to_component_keys(surface: str, segs: List[Tuple[int, int, str]]) -> List[str]:
    """
    Convert segments to component keys: substring(surface[a:b+1]) + "[" + kana + "]"
    Deduplicate within a word.
    """
    comps: List[str] = []
    seen = set()
    n = len(surface)
    for a, b, kana in segs:
        if a < 0 or b < 0 or a >= n or b >= n or b < a:
            continue
        substr = surface[a : b + 1]
        if not substr:
            continue
        key = f"{substr}[{kana}]"
        if key not in seen:
            seen.add(key)
            comps.append(key)
    return comps


# -----------------------
# Exact DP pieces (m <= 8)
# -----------------------
def mask_probs_for_component_set(comp_ids: Tuple[int, ...]) -> List[float]:
    """
    Worker-safe: uses globals. For fixed comp_ids (len m), compute p_mask over subsets 0..2^m-1.
    """
    m = len(comp_ids)
    if m == 0:
        return [1.0]

    word_mask: Dict[int, int] = {}
    inv = G_INVERTED_INDEX

    for i, cid in enumerate(comp_ids):
        bit = 1 << i
        for w in inv[cid]:
            word_mask[w] = word_mask.get(w, 0) | bit

    p_word = G_P_WORD
    p_mask = [0.0] * (1 << m)
    mass_nonzero = 0.0

    for w, mask in word_mask.items():
        pw = p_word[w]
        p_mask[mask] += pw
        mass_nonzero += pw

    p0 = 1.0 - mass_nonzero
    if p0 < 0.0:
        if p0 > -1e-12:
            p0 = 0.0
        else:
            p0 = max(0.0, p0)
    p_mask[0] = p0
    return p_mask


def expected_time_to_learn_all(p_mask: List[float], m: int, k: int) -> float:
    """
    Exact DP expected turns to reach all counts=k from counts=0.
    Complexity ~ 8^m when k=3, so we cap at m<=8.
    """
    if m == 0:
        return 0.0

    base = k + 1
    base_pows = [1] * m
    for i in range(1, m):
        base_pows[i] = base_pows[i - 1] * base

    n_states = base ** m
    goal = n_states - 1

    E = [0.0] * n_states
    E[goal] = 0.0

    # Precompute bits in each mask
    mask_bits: List[List[int]] = []
    for mask in range(1 << m):
        bits = []
        mm = mask
        idx = 0
        while mm:
            if mm & 1:
                bits.append(idx)
            mm >>= 1
            idx += 1
        mask_bits.append(bits)

    for s in range(goal - 1, -1, -1):
        tmp = s
        counts = [0] * m
        for i in range(m):
            counts[i] = tmp % base
            tmp //= base

        stay = 0.0
        acc = 0.0
        for mask, pm in enumerate(p_mask):
            if pm == 0.0:
                continue
            inc = 0
            for i in mask_bits[mask]:
                if counts[i] < k:
                    inc += base_pows[i]
            if inc == 0:
                stay += pm
            else:
                acc += pm * E[s + inc]

        denom = 1.0 - stay
        if denom <= 0.0:
            return float("inf")
        E[s] = (1.0 + acc) / denom

    return E[0]


def compute_exact_for_set(comp_set: Tuple[int, ...]) -> Tuple[Tuple[int, ...], float]:
    """
    Worker entrypoint: comp_set must have m<=MAX_M_EXACT
    """
    m = len(comp_set)
    p_mask = mask_probs_for_component_set(comp_set)
    burden = expected_time_to_learn_all(p_mask, m=m, k=G_K)
    return comp_set, burden


# -----------------------
# Heuristic for m>=9 (use heur_est_rms_turns)
# -----------------------
def load_weight_readings(path: str) -> Dict[str, float]:
    """
    Load _weightKanjireading.csv as mapping: component_key -> weight (float).
    We'll infer which column contains the key and which contains weight.
    """
    weights: Dict[str, float] = {}
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)
        if first is None:
            return weights

        # Detect whether first row is header
        has_alpha = any(any(ch.isalpha() for ch in cell) for cell in first)
        if has_alpha:
            cols = first
        else:
            # no header: treat first row as data, synthesize cols
            cols = [f"col{i}" for i in range(len(first))]
            # process first row as data below
            reader = iter([first] + list(reader))

        col_map = {name.strip().lower(): i for i, name in enumerate(cols)}

        # key column
        key_idx = None
        for name in POSSIBLE_KEY_COLS:
            if name in col_map:
                key_idx = col_map[name]
                break
        if key_idx is None:
            key_idx = 0  # fallback: first col

        # weight column: find a column literally called "weight" if present; else last column
        weight_idx = col_map.get("weight", None)
        if weight_idx is None:
            weight_idx = len(cols) - 1

        for row in reader:
            if not row or len(row) <= max(key_idx, weight_idx):
                continue
            key = row[key_idx].strip()
            if not key:
                continue
            try:
                w = float(row[weight_idx])
            except ValueError:
                continue
            weights[key] = w

    return weights


def heur_est_rms_turns(comp_keys: List[str], weight_map: Dict[str, float], k: int) -> float:
    """
    Heuristic estimate used for m>=9:

    - Convert component "weight" to a per-turn reinforcement probability:
        p_i ≈ weight_i / total_weight
    - Approx expected turns for k hits on component i alone:
        t_i ≈ k / p_i
    - Combine across components using RMS:
        heur_est_rms_turns = sqrt(sum_i t_i^2)

    This matches the prior "est_rms" behavior in 3.5_max8.py (kept intentionally).
    """
    vals = []
    for ck in comp_keys:
        w = weight_map.get(ck)
        if w is None:
            continue
        vals.append(w)

    if not vals:
        return float("inf")

    total_weight = sum(weight_map.values())
    if total_weight <= 0:
        return float("inf")

    s2 = 0.0
    for w in vals:
        p_i = w / total_weight
        if p_i <= 0:
            continue
        t = k / p_i
        s2 += t * t

    if s2 <= 0.0:
        return float("inf")
    return math.sqrt(s2)


# -----------------------
# Ranking
# -----------------------
def competition_ranks(values: List[float], ascending: bool = True) -> List[int]:
    """
    1224 competition ranking.

    Example: values sorted asc = [10, 20, 20, 30] -> ranks [1, 2, 2, 4]
    Returns ranks aligned with the input order.
    """
    n = len(values)
    pairs = list(enumerate(values))
    pairs.sort(key=lambda t: t[1], reverse=not ascending)

    ranks = [0] * n
    rank = 1
    i = 0
    while i < n:
        v = pairs[i][1]
        j = i + 1
        while j < n and pairs[j][1] == v:
            j += 1
        for k in range(i, j):
            ranks[pairs[k][0]] = rank
        rank += (j - i)
        i = j
    return ranks


def main() -> None:
    cwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(cwd)

    require_file(KANJIWORDS_CSV)
    require_file(FURIGANA_TXT)
    require_file(WEIGHT_READINGS_CSV)

    print(f"Using kanjiwords: {KANJIWORDS_CSV}")
    print(f"Using furigana : {FURIGANA_TXT}")
    print(f"Using weights  : {WEIGHT_READINGS_CSV}")
    print(f"THRESHOLD_LEARN = {THRESHOLD_LEARN}")
    print(f"CPU_CORES       = {CPU_CORES}")
    print(f"MAX_M_EXACT     = {MAX_M_EXACT}")
    print(f"DECIMAL_BURDEN  = {DECIMAL_BURDEN}")
    print()

    # 1) Read kanjiwords (rawrank must be stored as rawrank; p_word uses 1/rawrank separately)
    words: List[str] = []
    readings: List[str] = []
    rawranks: List[int] = []

    with open(KANJIWORDS_CSV, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("word") or "").strip()
            r = (row.get("reading") or "").strip()
            rr = (row.get("rawrank") or "").strip()
            if not w or not r or not rr:
                continue
            try:
                rawrank = int(rr)
            except ValueError:
                continue
            words.append(w)
            readings.append(r)
            rawranks.append(rawrank)

    n_words = len(words)
    print(f"Loaded {n_words} words.")

    # 2) Compute p(w) ∝ 1/rawrank  (DO NOT overwrite rawrank)
    inv = [0.0] * n_words
    Z = 0.0
    for i, rr in enumerate(rawranks):
        v = 1.0 / float(rr)
        inv[i] = v
        Z += v
    p_word = [v / Z for v in inv]

    # 3) Read furigana mapping: (surface, reading) -> component keys
    furigana_map: Dict[Tuple[str, str], List[str]] = {}
    with open(FURIGANA_TXT, encoding="utf-8") as f:
        for line in f:
            surface, reading, segs = parse_furigana_line(line)
            if not surface or not reading:
                continue
            comps = segments_to_component_keys(surface, segs)
            if comps:
                furigana_map[(surface, reading)] = comps
    print(f"Loaded {len(furigana_map)} furigana mappings.")

    # 4) Load component weights (needed for heuristic)
    weight_map = load_weight_readings(WEIGHT_READINGS_CSV)
    print(f"Loaded {len(weight_map)} component weights from _weightKanjireading.csv")

    # Seed IDs from weight file first (per your preference)
    comp_key_to_id: Dict[str, int] = {}
    for ck in weight_map.keys():
        comp_key_to_id[ck] = len(comp_key_to_id)

    word_comp_ids: List[List[int]] = [[] for _ in range(n_words)]
    word_comp_keys: List[List[str]] = [[] for _ in range(n_words)]
    missing_map = 0

    def get_comp_id(key: str) -> int:
        cid = comp_key_to_id.get(key)
        if cid is None:
            cid = len(comp_key_to_id)
            comp_key_to_id[key] = cid
        return cid

    for i in range(n_words):
        w = words[i]
        r = readings[i]
        comps = furigana_map.get((w, r))
        if not comps:
            missing_map += 1
            comps = [f"{w}[{r}]"]  # fallback component for unmapped

        # Deduplicate within the word
        seen = set()
        keys: List[str] = []
        ids: List[int] = []
        for ck in comps:
            if ck in seen:
                continue
            seen.add(ck)
            keys.append(ck)
            ids.append(get_comp_id(ck))

        word_comp_keys[i] = keys
        word_comp_ids[i] = ids

    print(f"Global component vocab size (after seeding+new): {len(comp_key_to_id)}")
    print(f"Words missing furigana mapping (fallback used): {missing_map}")

    # 5) Build inverted index
    n_comp = len(comp_key_to_id)
    inverted_index: List[List[int]] = [[] for _ in range(n_comp)]
    edges = 0
    for w_id, comps in enumerate(word_comp_ids):
        for cid in comps:
            inverted_index[cid].append(w_id)
            edges += 1
    print(f"Inverted index built. Total edges: {edges}")

    # 6) Build unique component-sets and split into exact vs long
    word_key: List[Tuple[int, ...]] = [tuple() for _ in range(n_words)]
    exact_sets: List[Tuple[int, ...]] = []
    all_sets = set()

    for w_id in range(n_words):
        key = tuple(sorted(word_comp_ids[w_id]))
        word_key[w_id] = key
        if key not in all_sets:
            all_sets.add(key)
            if len(key) <= MAX_M_EXACT:
                exact_sets.append(key)

    max_m = max((len(s) for s in all_sets), default=0)
    print(f"Unique component-sets total: {len(all_sets)} (max m = {max_m})")
    print(f"Exact sets (m<= {MAX_M_EXACT}): {len(exact_sets)}")
    print(f"Long sets  (m>= {MAX_M_EXACT+1}): {len(all_sets) - len(exact_sets)}")

    # 7) Parallel compute exact burdens for exact_sets only
    cache_exact: Dict[Tuple[int, ...], float] = {}

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=CPU_CORES,
        initializer=_init_worker,
        initargs=(inverted_index, p_word, THRESHOLD_LEARN),
    ) as pool:
        processed = 0
        for comp_set, burden in pool.imap_unordered(compute_exact_for_set, exact_sets, chunksize=CHUNKSIZE):
            cache_exact[comp_set] = burden
            processed += 1
            if processed % 2000 == 0:
                print(f"Exact computed {processed}/{len(exact_sets)} sets...")

    print("Exact DP done.")

    # 8) Compute burden per word (exact for m<=8; heuristic RMS for m>=9), round, then rank
    n_components: List[int] = [0] * n_words
    burden_rounded: List[float] = [float("inf")] * n_words

    for i in range(n_words):
        key = word_key[i]
        m = len(key)
        n_components[i] = m
        if m <= MAX_M_EXACT:
            b = cache_exact[key]
        else:
            b = heur_est_rms_turns(comp_keys=word_comp_keys[i], weight_map=weight_map, k=THRESHOLD_LEARN)

        if math.isfinite(b):
            burden_rounded[i] = round(b, DECIMAL_BURDEN)
        else:
            burden_rounded[i] = float("inf")

    rank_on_raw = competition_ranks([float(x) for x in rawranks], ascending=True)
    rank_on_burden = competition_ranks(burden_rounded, ascending=True)

    # 9) Write main output (NO jukugodifficultysheet.csv anymore)
    with open(OUT_MAIN, "w", encoding="utf-8", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["word", "reading", "n_components", "rank", "kanji_rank", "rawrank", "kanji_burden_turns"])
        for i in range(n_words):
            b = burden_rounded[i]
            b_str = "inf" if not math.isfinite(b) else f"{b:.{DECIMAL_BURDEN}f}"
            wcsv.writerow([
                words[i],
                readings[i],
                n_components[i],
                rank_on_raw[i],
                rank_on_burden[i],
                rawranks[i],
                b_str,
            ])

    print(f"Wrote: {OUT_MAIN}")


if __name__ == "__main__":
    main()
