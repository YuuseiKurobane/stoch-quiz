# kanjiwords_jmdictslice_from_raw_v4_competition_nolatin.py
#
# Output : kanjiwords_jmdictslice.csv
# Inputs : JmdictFurigana.txt , term_meta_bank_1.json
#
# Output columns:
#   word, reading, freq, rawrank
#
# Filters (drop entries):
# 1) displayValue contains '㋕' (kana-frequency marker)
# 2) word not present in JmdictFurigana headwords (text before first '|')
# 3) word (surface form) contains any Latin letter:
#    - ASCII A–Z / a–z
#    - Full-width Latin FF21–FF3A, FF41–FF5A
#    - (Plus any Unicode letter whose name contains 'LATIN', e.g. accented Latin)
#
# Ranking:
# - Competition ranking (aka 1224 ranking) after filtering.
#   Example: 3716, 3717, 3717, 3719 ...
#
# Warnings (max 50):
# - If multiple kept rows share the same rawrank, print a warning including displayValue exactly as in JSON.
#
# Assumption:
# - term_meta_bank_1.json freq entries are in nondecreasing rawrank order. If not, script exits.

from __future__ import annotations

import csv
import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Generator, Optional, Set, List


INPUT_JSON = "term_meta_bank_1.json"
INPUT_JMDICT_FURIGANA = "JmdictFurigana.txt"
OUTPUT_CSV = "kanjiwords_jmdictslice.csv"

KANA_MARK = "㋕"
MAX_WARNINGS = 50


@dataclass(frozen=True)
class ParsedRow:
    word: str
    reading: str
    rawrank: int
    display_value: str


def iter_top_level_json_array(path: str) -> Generator[Any, None, None]:
    decoder = json.JSONDecoder()
    buf = ""
    pos = 0
    started = False
    ended = False

    with open(path, "r", encoding="utf-8") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            buf += chunk

            while True:
                while pos < len(buf) and buf[pos].isspace():
                    pos += 1

                if not started:
                    if pos < len(buf) and buf[pos] == "[":
                        started = True
                        pos += 1
                    else:
                        break

                while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ","):
                    pos += 1

                if pos >= len(buf):
                    break

                if buf[pos] == "]":
                    ended = True
                    pos += 1
                    break

                try:
                    obj, next_pos = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    break

                yield obj
                pos = next_pos

            if pos > 1024 * 1024:
                buf = buf[pos:]
                pos = 0

    if not started:
        raise ValueError("Input does not start with a JSON array '['.")
    if not ended:
        tail = buf[pos:].strip()
        if tail != "" and tail != "]":
            raise ValueError("Input JSON array did not terminate properly with ']'.")


def load_jmdict_headwords(path: str) -> Set[str]:
    headwords: Set[str] = set()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            bar = line.find("|")
            if bar == -1:
                continue
            head = line[:bar]
            if head:
                headwords.add(head)
    return headwords


def parse_entry_to_row(entry: Any) -> Optional[ParsedRow]:
    if not isinstance(entry, list) or len(entry) < 3:
        return None
    if entry[1] != "freq":
        return None

    word = entry[0]
    meta = entry[2]
    if not isinstance(word, str) or not isinstance(meta, dict):
        return None

    # Shape A: ["ふたり","freq",{"value":4813,"displayValue":"4813㋕"}]
    if "value" in meta and "displayValue" in meta:
        try:
            rawrank = int(meta["value"])
        except Exception:
            return None
        display_value = str(meta["displayValue"])
        reading = word
        return ParsedRow(word=word, reading=reading, rawrank=rawrank, display_value=display_value)

    # Shape B: ["明白","freq",{"reading":"めいはく","frequency":{"value":123,"displayValue":"123"}}]
    if "reading" in meta and "frequency" in meta and isinstance(meta["frequency"], dict):
        freq = meta["frequency"]
        if "value" not in freq or "displayValue" not in freq:
            return None
        try:
            rawrank = int(freq["value"])
        except Exception:
            return None
        display_value = str(freq["displayValue"])
        reading = str(meta["reading"])
        return ParsedRow(word=word, reading=reading, rawrank=rawrank, display_value=display_value)

    return None


def is_kana_marked(display_value: str) -> bool:
    return KANA_MARK in display_value


def is_latin_letter_char(ch: str) -> bool:
    o = ord(ch)

    # ASCII A-Z, a-z
    if 0x41 <= o <= 0x5A or 0x61 <= o <= 0x7A:
        return True

    # Full-width Latin A–Z, a–z
    if 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
        return True

    # Any other Unicode LATIN letter (e.g., é, ü, etc.)
    # Only treat as Latin if it's a letter and its Unicode name contains "LATIN".
    # (unicodedata.name raises ValueError for unnamed codepoints)
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return False

    if "LATIN" in name:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            return True

    return False


def contains_any_latin_letter(s: str) -> bool:
    for ch in s:
        if is_latin_letter_char(ch):
            return True
    return False


def main() -> None:
    if not os.path.exists(INPUT_JMDICT_FURIGANA):
        print(f"ERROR: Missing {INPUT_JMDICT_FURIGANA}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(INPUT_JSON):
        print(f"ERROR: Missing {INPUT_JSON}", file=sys.stderr)
        sys.exit(1)

    headwords = load_jmdict_headwords(INPUT_JMDICT_FURIGANA)
    print(f"Headwords loaded: {len(headwords)}", file=sys.stderr)

    warnings_printed = 0

    # Competition rank counter after filtering
    next_rank_value = 1

    # Grouping by rawrank (assumes nondecreasing rawrank in JSON freq entries)
    current_rawrank: Optional[int] = None
    current_group_rows: List[ParsedRow] = []  # kept rows of this rawrank
    last_seen_rawrank_any: Optional[int] = None

    def flush_group() -> None:
        nonlocal next_rank_value, current_group_rows
        if not current_group_rows:
            return
        assigned_rank = next_rank_value
        for r in current_group_rows:
            writer.writerow([r.word, r.reading, assigned_rank, r.rawrank])
        next_rank_value += len(current_group_rows)
        current_group_rows = []

    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["word", "reading", "freq", "rawrank"])

        for entry in iter_top_level_json_array(INPUT_JSON):
            row = parse_entry_to_row(entry)
            if row is None:
                continue

            # Ordering sanity-check for freq entries
            if last_seen_rawrank_any is not None and row.rawrank < last_seen_rawrank_any:
                print(
                    f"ERROR: rawrank decreased ({last_seen_rawrank_any} -> {row.rawrank}). "
                    f"This script assumes nondecreasing rawrank order in term_meta_bank_1.json.",
                    file=sys.stderr,
                )
                sys.exit(2)
            last_seen_rawrank_any = row.rawrank

            # Group boundary on rawrank change (even if the new item later gets filtered out)
            if current_rawrank is None:
                current_rawrank = row.rawrank
            elif row.rawrank != current_rawrank:
                flush_group()
                current_rawrank = row.rawrank

            # Filters
            if is_kana_marked(row.display_value):
                continue
            if row.word not in headwords:
                continue
            if contains_any_latin_letter(row.word):
                continue

            # Duplicate warning within this rawrank group (kept rows only)
            if current_group_rows and warnings_printed < MAX_WARNINGS:
                print(
                    f"WARNING duplicate rawrank={row.rawrank} displayValue={row.display_value} "
                    f"word={row.word} reading={row.reading}",
                    file=sys.stderr,
                )
                warnings_printed += 1

            current_group_rows.append(row)

        flush_group()

    print(f"Done. Wrote: {OUTPUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
