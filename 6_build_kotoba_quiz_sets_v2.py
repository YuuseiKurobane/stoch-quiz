#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build Kotoba quiz-bot upload CSVs (one per bucket) from:
  - quiz_buckets_strong_Final.csv (bucket assignments for (word, reading))
  - JMdict (XML; typically "JMdict" or "JMdict.gz")
  - new_quiz.csv (template header + default Instructions / Render as)

Outputs:
  out_kotoba_sets/S0.csv ... S15.csv and L2.csv ... L15.csv
  out_kotoba_sets/_warnings.txt

Each row:
  Question      = word (surface)
  Answers       = comma-separated list of all valid readings from JMdict for that surface
  Comment       = up to 3 English glosses (shortened)
  Instructions  = from new_quiz.csv template (row 1)
  Render as     = from new_quiz.csv template (row 1)

Behavior notes:
  - "valid readings" respect JMdict <re_restr> (reading applies only to some keb)
  - glosses use English only (xml:lang absent or "eng")
  - meaning prioritization: prefer senses restricted to the target (word/reading) via <stagk>/<stagr>.
    If none match, fall back to general senses, then to any English gloss.
  - warnings are printed when:
      * (word, reading) not found in JMdict at all
      * meanings had to fall back (no reading-specific sense)
      * target reading not among the valid readings returned (still outputs all readings)

You probably want to point JMDICT_PATH at your full JMdict file (not the 10k-line sample).
"""

from __future__ import annotations

import csv
import gzip
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
import xml.etree.ElementTree as ET

import pandas as pd

# ----------------------------- USER CONFIG -----------------------------

# Put these 3 inputs in the same folder as this script, or edit the paths.
QUIZ_BUCKETS_PATH = Path("quiz_buckets_strong_Final.csv")
NEW_QUIZ_TEMPLATE_PATH = Path("new_quiz.csv")

# Full JMdict file path (XML). Common filenames: "JMdict", "JMdict.xml", "JMdict.gz"
JMDICT_PATH = Path("JMdict")

# Output folder
OUT_DIR = Path("out_kotoba_sets")

# Comment shortening controls
MAX_GLOSSES = 3
MAX_GLOSS_CHARS = 80       # per gloss
MAX_COMMENT_CHARS = 220    # after joining

# If you want to exclude readings marked <re_nokanji/>, set this True.
# For reading quizzes, you usually still want them accepted, so default False.
EXCLUDE_RE_NOKANJI = False

# ----------------------------------------------------------------------


@dataclass
class ReadingInfo:
    reb: str
    applies_to: Optional[Set[str]]  # None => applies to all kebs in entry (or kana-only entry)
    no_kanji: bool                  # re_nokanji present


@dataclass
class SenseInfo:
    stagk: Set[str]
    stagr: Set[str]
    glosses_eng: List[str]          # in order of appearance


@dataclass
class EntryInfo:
    ent_seq: str
    kebs: Set[str]
    readings: List[ReadingInfo]
    senses: List[SenseInfo]


def _open_maybe_gzip(path: Path) -> io.BufferedReader:
    """
    Open plain text or .gz. Returns a binary file-like object.
    """
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rb")
    return open(path, "rb")


def _get_xml_lang(elem: ET.Element) -> str:
    # xml:lang is in the XML namespace
    # In ElementTree, it appears as '{http://www.w3.org/XML/1998/namespace}lang'
    return elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "eng")


def _clean_gloss(s: str) -> str:
    s = " ".join((s or "").strip().split())
    # Remove placeholder-only glosses seen in some entries
    if s in {"*", "_"}:
        return ""
    return s


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    if max_chars <= 1:
        return s[:max_chars]
    return s[: max_chars - 1].rstrip() + "…"


def _dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for x in items:
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def parse_jmdict_subset(jmdict_path: Path, needed_words: Set[str]) -> Dict[str, List[EntryInfo]]:
    """
    Stream-parse JMdict and keep only entries relevant to needed_words.

    Keys in returned dict:
      - every keb that appears in needed_words
      - for kana-only entries (no keb), every reb that appears in needed_words
    """
    word_to_entries: Dict[str, List[EntryInfo]] = {}

    with _open_maybe_gzip(jmdict_path) as f:
        # iterparse over bytes; ET will handle UTF-8 bytes
        context = ET.iterparse(f, events=("end",))
        for event, elem in context:
            if elem.tag != "entry":
                continue

            ent_seq = (elem.findtext("ent_seq") or "").strip()

            kebs: Set[str] = set()
            for k_ele in elem.findall("k_ele"):
                keb = (k_ele.findtext("keb") or "").strip()
                if keb:
                    kebs.add(keb)

            readings: List[ReadingInfo] = []
            for r_ele in elem.findall("r_ele"):
                reb = (r_ele.findtext("reb") or "").strip()
                if not reb:
                    continue
                restr_elems = r_ele.findall("re_restr")
                applies_to = None if len(restr_elems) == 0 else { (x.text or "").strip() for x in restr_elems if (x.text or "").strip() }
                no_kanji = r_ele.find("re_nokanji") is not None
                readings.append(ReadingInfo(reb=reb, applies_to=applies_to, no_kanji=no_kanji))

            senses: List[SenseInfo] = []
            for s_ele in elem.findall("sense"):
                stagk = { (x.text or "").strip() for x in s_ele.findall("stagk") if (x.text or "").strip() }
                stagr = { (x.text or "").strip() for x in s_ele.findall("stagr") if (x.text or "").strip() }

                glosses_eng: List[str] = []
                for g in s_ele.findall("gloss"):
                    lang = _get_xml_lang(g)
                    if lang != "eng":
                        continue
                    gloss = _clean_gloss(g.text or "")
                    if gloss:
                        glosses_eng.append(gloss)

                senses.append(SenseInfo(stagk=stagk, stagr=stagr, glosses_eng=glosses_eng))

            entry = EntryInfo(ent_seq=ent_seq, kebs=kebs, readings=readings, senses=senses)

            # Keep only relevant mappings
            mapped_any = False

            for keb in kebs:
                if keb in needed_words:
                    word_to_entries.setdefault(keb, []).append(entry)
                    mapped_any = True

            if not kebs:
                # kana-only entry: map by reb(s)
                for r in readings:
                    if r.reb in needed_words:
                        word_to_entries.setdefault(r.reb, []).append(entry)
                        mapped_any = True

            # Important: clear to keep memory usage low
            elem.clear()

    return word_to_entries


def valid_readings_for_word(entry: EntryInfo, word: str) -> List[str]:
    """
    Return list of readings (reb) that are valid for 'word' in this entry,
    respecting re_restr.
    """
    out: List[str] = []
    for r in entry.readings:
        if EXCLUDE_RE_NOKANJI and r.no_kanji:
            continue
        if r.applies_to is None:
            # applies to all kebs (or to kana-only entry)
            if entry.kebs and word not in entry.kebs:
                # This entry doesn't actually define this keb, but may be mapped via kana-only;
                # keep it permissive.
                out.append(r.reb)
            else:
                out.append(r.reb)
        else:
            if word in r.applies_to:
                out.append(r.reb)
    return out


def pick_best_entry(entries: List[EntryInfo], word: str, target_reading: str) -> Tuple[Optional[EntryInfo], bool]:
    """
    Choose the best entry for (word, target_reading).

    Returns: (EntryInfo or None, target_reading_in_valid_readings_bool)
    """
    best = None
    best_score = -10**9
    best_has_target = False

    for e in entries:
        vreads = set(valid_readings_for_word(e, word))
        has_target = target_reading in vreads

        score = 0
        if word in e.kebs:
            score += 10
        if not e.kebs:
            score += 5  # kana-only entry
        if has_target:
            score += 20
        # mild preference: fewer keb variants (more "direct")
        score -= max(0, len(e.kebs) - 1)

        if score > best_score:
            best_score = score
            best = e
            best_has_target = has_target

    return best, best_has_target


def pick_english_glosses(entry: EntryInfo, word: str, target_reading: str) -> Tuple[List[str], bool]:
    """
    Pick up to MAX_GLOSSES English glosses for the entry, prioritizing senses
    restricted to (word/reading) if available.

    Returns: (glosses, used_fallback_bool)
    """
    # 1) Reading/word-specific senses
    specific: List[str] = []
    general: List[str] = []
    all_eng: List[str] = []

    for s in entry.senses:
        if s.glosses_eng:
            all_eng.extend(s.glosses_eng)

        is_specific = False
        # If stagk exists, sense is restricted to listed keb(s).
        if s.stagk and word in s.stagk:
            is_specific = True
        # If stagr exists, sense is restricted to listed reading(s).
        if s.stagr and target_reading in s.stagr:
            is_specific = True

        if is_specific:
            specific.extend(s.glosses_eng)
        else:
            # "general" means: sense not restricted by any stagk/stagr
            if not s.stagk and not s.stagr:
                general.extend(s.glosses_eng)

    specific = _dedupe_preserve_order(specific)
    general = _dedupe_preserve_order(general)
    all_eng = _dedupe_preserve_order(all_eng)

    used_fallback = False
    chosen = specific
    if not chosen:
        used_fallback = True
        chosen = general if general else all_eng

    chosen = chosen[:MAX_GLOSSES]
    chosen = [_truncate(x, MAX_GLOSS_CHARS) for x in chosen]
    return chosen, used_fallback


def main() -> int:
    # Load Kotoba template
    template_df = pd.read_csv(NEW_QUIZ_TEMPLATE_PATH)
    if template_df.shape[0] < 1:
        raise SystemExit(f"Template file has no rows: {NEW_QUIZ_TEMPLATE_PATH}")
    required_cols = ["Question", "Answers", "Comment", "Instructions", "Render as"]
    for c in required_cols:
        if c not in template_df.columns:
            raise SystemExit(f"Template missing column '{c}'. Found: {list(template_df.columns)}")

    default_instructions = str(template_df.loc[0, "Instructions"])
    default_render_as = str(template_df.loc[0, "Render as"])

    # Load buckets
    buckets_df = pd.read_csv(QUIZ_BUCKETS_PATH)
    if "bucket" not in buckets_df.columns:
        raise SystemExit("quiz_buckets_strong_Final.csv must contain a 'bucket' column.")
    for c in ("word", "reading"):
        if c not in buckets_df.columns:
            raise SystemExit(f"quiz_buckets_strong_Final.csv missing column '{c}'.")

    # Clean + unique
    buckets_df["word"] = buckets_df["word"].astype(str).str.strip()
    buckets_df["reading"] = buckets_df["reading"].astype(str).str.strip()

    pairs = buckets_df[["bucket", "word", "reading"]].drop_duplicates()
    needed_words: Set[str] = set(pairs["word"].tolist())

    # Parse JMdict (subset)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    warnings_path = OUT_DIR / "_warnings.txt"
    warn_lines: List[str] = []

    if not JMDICT_PATH.exists():
        raise SystemExit(
            f"JMdict file not found: {JMDICT_PATH}\n"
            f"Edit JMDICT_PATH at the top of the script to point to your full JMdict (XML)."
        )

    print(f"Parsing JMdict (subset) from: {JMDICT_PATH}")
    print(f"Unique surfaces needed: {len(needed_words)}")
    word_to_entries = parse_jmdict_subset(JMDICT_PATH, needed_words)
    print(f"JMdict entries kept for needed surfaces: {sum(len(v) for v in word_to_entries.values())}")

    # Generate per-bucket outputs
    buckets = sorted(pairs["bucket"].unique().tolist(), key=lambda x: (x[0], int(x[1:]) if x[1:].isdigit() else 0))

    for bucket in buckets:
        sub = pairs[pairs["bucket"] == bucket]
        out_rows: List[Dict[str, str]] = []

        for _, r in sub.iterrows():
            word = r["word"]
            target_reading = r["reading"]

            entries = word_to_entries.get(word, [])
            if not entries:
                warn_lines.append(f"[MISSING] {bucket}: {word} [{target_reading}] not found in JMdict")
                # still output something so you can see it and fix later
                out_rows.append({
                    "Question": word,
                    "Answers": target_reading,
                    "Comment": "",
                    "Instructions": default_instructions,
                    "Render as": default_render_as,
                })
                continue

            best_entry, has_target = pick_best_entry(entries, word, target_reading)
            if best_entry is None:
                warn_lines.append(f"[MISSING] {bucket}: {word} [{target_reading}] best_entry=None (unexpected)")
                out_rows.append({
                    "Question": word,
                    "Answers": target_reading,
                    "Comment": "",
                    "Instructions": default_instructions,
                    "Render as": default_render_as,
                })
                continue

            # Answers must accept ANY valid JMdict reading for this surface, even if not present in quiz_buckets.
            # So we take the union of valid readings across all entries that define this word (respecting <re_restr>).
            vreads_union: List[str] = []
            for e in entries:
                vreads_union.extend(valid_readings_for_word(e, word))
            vreads = _dedupe_preserve_order(vreads_union)
            if not vreads:
                warn_lines.append(f"[WARN] {bucket}: {word} [{target_reading}] found entry {best_entry.ent_seq} but no valid readings (re_restr mismatch?)")
                vreads = [target_reading]

            target_in_union = target_reading in vreads
            if not target_in_union:
                warn_lines.append(f"[WARN] {bucket}: {word} [{target_reading}] not among valid readings: {','.join(vreads)} (best-entry {best_entry.ent_seq})")

            glosses, used_fallback = pick_english_glosses(best_entry, word, target_reading)
            if used_fallback:
                warn_lines.append(f"[FALLBACK] {bucket}: {word} [{target_reading}] no reading-specific sense; used general/any (entry {best_entry.ent_seq})")

            comment = "; ".join(glosses)
            comment = _truncate(comment, MAX_COMMENT_CHARS)

            out_rows.append({
                "Question": word,
                "Answers": ",".join(vreads),
                "Comment": comment,
                "Instructions": default_instructions,
                "Render as": default_render_as,
            })

        out_path = OUT_DIR / f"{bucket}.csv"
        pd.DataFrame(out_rows, columns=required_cols).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Wrote {bucket}: {len(out_rows)} rows -> {out_path}")

    warnings_path.write_text("\n".join(warn_lines) + ("\n" if warn_lines else ""), encoding="utf-8")
    print(f"\nWarnings written to: {warnings_path}")
    if warn_lines:
        print(f"Warnings count: {len(warn_lines)} (open _warnings.txt)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
