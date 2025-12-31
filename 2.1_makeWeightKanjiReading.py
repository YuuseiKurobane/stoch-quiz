#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.1_makeWeightKanjiReading.py

Build weightKanjiReading_Final.csv with *proper fallback logic*.

Definition (consistent with your immersion story & your current pipeline):
- You have a word list with (word, reading, rawrank).
- Each turn draws exactly 1 word token with probability p(word) ∝ 1/rawrank (normalized).
- Each token yields 1 or more kanji-component-readings (component keys) from JmdictFurigana.
- Within a token, a component contributes at most once (presence/absence; dedupe).
- If (word, reading) is missing in furigana mapping OR yields no components, fallback to:
      component key = f"{word}[{reading}]"
- Component per-turn appearance probability:
      p(component) = Σ_{word contains component} p(word)
- Output "weight" is:
      weight = 1 / p(component)
  (interpretable as expected turns to see the component once, under the presence/absence model)

Outputs:
- weightKanjiReading_Final.csv columns:
    kanjireading, rank, weight
  where rank=1 is the smallest weight (most frequent/most likely per turn).

Inputs expected in the same folder as this script (or adjust paths below):
- kanjiwords_jmdictslice.csv
- JmdictFurigana_For_ChatGPT.txt
"""

from __future__ import annotations
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Set


# -----------------------
# Config
# -----------------------
KANJIWORDS_CSV = "kanjiwords_jmdictslice.csv"
FURIGANA_TXT   = "JmdictFurigana.txt" # Not for chatgpt. Use actual file
OUTPUT_CSV     = "weightKanjiReading_Final.csv"

# If True, also write a small stats JSON next to the output
WRITE_STATS_JSON = True
STATS_JSON = "weightKanjiReading_Final.stats.json"

# Numerical formatting
WEIGHT_DECIMALS = 12  # keep high precision; you can lower later


def parse_furigana_line(line: str) -> Tuple[str, str, List[Tuple[int, int, str]]]:
    """
    Format in JmdictFurigana_For_ChatGPT.txt:
      surface|reading|segs
    segs are separated by ';'
    each seg: 'idx:kana' or 'a-b:kana'  (0-indexed inclusive)
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
    Convert segments (a,b,kana) -> component key = surface[a:b+1] + "[" + kana + "]"
    Dedupe within the token (presence/absence).
    """
    out: List[str] = []
    seen: Set[str] = set()
    L = len(surface)

    for a, b, kana in segs:
        if a < 0 or b < a or b >= L:
            continue
        substr = surface[a:b+1]
        if not substr:
            continue
        key = f"{substr}[{kana}]"
        if key not in seen:
            seen.add(key)
            out.append(key)

    return out


def load_furigana_map(path: str) -> Dict[Tuple[str, str], List[str]]:
    """
    Map (surface, reading) -> list of component keys (deduped).
    """
    mp: Dict[Tuple[str, str], List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            surface, reading, segs = parse_furigana_line(line)
            if not surface or not reading:
                continue
            comps = segments_to_component_keys(surface, segs)
            if comps:
                mp[(surface, reading)] = comps
    return mp


def load_words(path: str) -> Tuple[List[str], List[str], List[int]]:
    words: List[str] = []
    reads: List[str] = []
    rawranks: List[int] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        required = {"word", "reading", "rawrank"}
        if not required.issubset(set(rdr.fieldnames or [])):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")

        for row in rdr:
            w = (row["word"] or "").strip()
            r = (row["reading"] or "").strip()
            rr = (row["rawrank"] or "").strip()
            if not w or not r or not rr:
                continue
            try:
                rawrank = int(rr)
            except ValueError:
                continue
            if rawrank <= 0:
                continue
            words.append(w)
            reads.append(r)
            rawranks.append(rawrank)

    return words, reads, rawranks


def compute_p_word(rawranks: List[int]) -> List[float]:
    """
    p(word) ∝ 1/rawrank, normalized.
    """
    inv = [1.0 / float(rr) for rr in rawranks]
    Z = sum(inv)
    return [v / Z for v in inv]


def main() -> None:
    base = Path(__file__).resolve().parent
    in_words = base / KANJIWORDS_CSV
    in_furi  = base / FURIGANA_TXT
    out_csv  = base / OUTPUT_CSV

    if not in_words.exists():
        raise FileNotFoundError(str(in_words))
    if not in_furi.exists():
        raise FileNotFoundError(str(in_furi))

    print("Loading furigana mapping...")
    furimap = load_furigana_map(str(in_furi))
    print(f"Loaded {len(furimap)} furigana mappings.")

    print("Loading word list...")
    words, reads, rawranks = load_words(str(in_words))
    print(f"Loaded {len(words)} rows from {KANJIWORDS_CSV}")

    print("Computing p(word) ...")
    p_word = compute_p_word(rawranks)

    # Compute p(component) using presence/absence per token
    p_comp: Dict[str, float] = defaultdict(float)
    missing = 0
    total_components_emitted = 0  # for average components per token

    for wi, (w, r) in enumerate(zip(words, reads)):
        comps = furimap.get((w, r))
        if not comps:
            missing += 1
            comps = [f"{w}[{r}]"]  # fallback (dedup implicitly)
        total_components_emitted += len(comps)
        pw = p_word[wi]
        for c in comps:
            p_comp[c] += pw

    print(f"Unique component keys: {len(p_comp)}")
    print(f"Fallback used for {missing} words.")
    print(f"Weighted average #components per token (presence keys): {total_components_emitted / max(1, len(words)):.6f}")

    # Convert to weights
    rows = []
    for key, p in p_comp.items():
        if p <= 0.0:
            continue
        w = 1.0 / p
        rows.append((key, w))

    # Sort by weight ascending (most probable first)
    rows.sort(key=lambda x: x[1])

    print("Writing output CSV...")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["kanjireading", "rank", "weight"])
        for idx, (key, w) in enumerate(rows, 1):
            wr.writerow([key, idx, f"{w:.{WEIGHT_DECIMALS}f}"])

    print(f"Wrote: {out_csv}")

    if WRITE_STATS_JSON:
        stats = {
            "n_words": len(words),
            "n_furigana_mappings": len(furimap),
            "n_components": len(rows),
            "fallback_words": missing,
            "min_weight": rows[0][1] if rows else None,
            "max_weight": rows[-1][1] if rows else None,
        }
        with open(base / STATS_JSON, "w", encoding="utf-8") as f:
            import json
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"Wrote: {base / STATS_JSON}")


if __name__ == "__main__":
    main()
