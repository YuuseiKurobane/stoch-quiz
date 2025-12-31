#!/usr/bin/env python3
"""
make_quiz_commands.py

Inputs (in the same folder by default):
  - quiz_pass_fail_marks.csv
  - manual-input-quiz.csv

Output:
  - 15_quiz_commands.txt

CSV formats:

quiz_pass_fail_marks.csv columns:
  level,total_questions,pass_threshold

manual-input-quiz.csv columns:
  level,set,items,weight

Notes / requirements implemented:
  - Weighted percentage per set within each level:
        eff = items * weight
        pct_raw = eff / sum(eff) * 100
  - Percentages are based on 3-decimal strings (so they round-trip through float parsing).
  - We then nudge by <= 0.002 percentage points per set (in 0.001 steps for non-last sets)
    to force the *Python-float* sum of percentages to be exactly 100.0.
  - Set names map to Kotoba sets as: "stoch" + lowercase(set), e.g. S0 -> stochs0, L12 -> stochl12
  - Pass/fail:
        mmq = total_questions - pass_threshold
        accuracy = pass_threshold / total_questions * 100  (header rounded to 1 decimal)
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SetRow:
    level: int
    set_name: str
    items: int
    weight: float


@dataclass(frozen=True)
class PassFailRow:
    level: int
    total_questions: int
    pass_threshold: int


def parse_set_key(set_name: str) -> Tuple[int, int, str]:
    """
    Sort key: S-series first, then L-series, numeric ascending.
    """
    s = set_name.strip()
    if not s:
        return (2, 10**9, s)
    letter = s[0].upper()
    num_part = s[1:]
    try:
        num = int(num_part)
    except ValueError:
        num = 10**9
    group = 0 if letter == "S" else (1 if letter == "L" else 2)
    return (group, num, s)


def read_manual_input(path: Path) -> Dict[int, List[SetRow]]:
    by_level: Dict[int, List[SetRow]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"level", "set", "items", "weight"}
        if set(r.fieldnames or []) != required:
            # allow extra columns, but require these
            missing = required - set(r.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name}: missing columns: {sorted(missing)} (got {r.fieldnames})")
        for row in r:
            level = int(row["level"])
            set_name = str(row["set"]).strip()
            items = int(row["items"])
            weight = float(row["weight"])
            by_level.setdefault(level, []).append(SetRow(level, set_name, items, weight))

    # ensure deterministic order
    for lvl, rows in by_level.items():
        by_level[lvl] = sorted(rows, key=lambda x: parse_set_key(x.set_name))
    return by_level


def read_pass_fail(path: Path) -> Dict[int, PassFailRow]:
    by_level: Dict[int, PassFailRow] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"level", "total_questions", "pass_threshold"}
        if set(r.fieldnames or []) != required:
            missing = required - set(r.fieldnames or [])
            if missing:
                raise ValueError(f"{path.name}: missing columns: {sorted(missing)} (got {r.fieldnames})")
        for row in r:
            lvl = int(row["level"])
            total_questions = int(row["total_questions"])
            pass_threshold = int(row["pass_threshold"])
            by_level[lvl] = PassFailRow(lvl, total_questions, pass_threshold)
    return by_level


def sequential_sum(values: List[float]) -> float:
    s = 0.0
    for v in values:
        s += v
    return s


def compute_percent_strings(effs: List[float]) -> Tuple[List[str], List[float], List[float]]:
    """
    Returns:
      - pct_strs: strings to embed inside "(...%)"
      - final_vals: floats that pct_strs parse back into (Python float)
      - base_vals: base 3-decimal floats before nudging
    """
    if not effs:
        raise ValueError("No sets in level")

    total = sum(effs)
    raw = [(e / total) * 100.0 for e in effs]

    # Base values as 3-decimal strings -> float, so formatting/parsing round-trips.
    base_vals = [float(f"{p:.3f}") for p in raw]

    if len(base_vals) == 1:
        # single set: force exactly 100
        final_vals = [100.0]
        pct_strs = ["100.000"]
        return pct_strs, final_vals, base_vals

    prev = base_vals[:-1].copy()
    base_last = base_vals[-1]

    last0 = 100.0 - sequential_sum(prev)
    diff_last = last0 - base_last

    if abs(diff_last) <= 0.002 + 1e-12:
        final_prev = prev
    else:
        # Move last into the allowed band by nudging prev-sum in 0.001 steps.
        excess = abs(diff_last) - 0.002
        steps = int(math.ceil(excess / 0.001 - 1e-12))  # smallest integer steps to cover excess
        delta_total = steps * 0.001
        direction = 1.0 if diff_last > 0 else -1.0  # + means increase prev sum (decrease last)

        remaining = delta_total
        final_prev = prev.copy()

        # Distribute from the end; each set may be nudged up to 0.002.
        for i in range(len(final_prev) - 1, -1, -1):
            if remaining <= 1e-12:
                break

            take = min(0.002, remaining)
            # Prefer 0.002 then 0.001, keeping 3 decimals.
            if take >= 0.0015 and take >= 0.002 - 1e-12:
                take_q = 0.002
            else:
                take_q = 0.001

            if take_q > remaining + 1e-12:
                take_q = remaining

            final_prev[i] = float(f"{(final_prev[i] + direction * take_q):.3f}")
            remaining -= take_q

        if remaining > 1e-9:
            raise RuntimeError(f"Could not distribute adjustment (remaining={remaining})")

    final_last = 100.0 - sequential_sum(final_prev)
    final_vals = final_prev + [final_last]

    # Sanity checks
    if sequential_sum(final_vals) != 100.0:
        # Force last to make the float sum exactly 100.0 in the same sequential order.
        forced_last = 100.0 - sequential_sum(final_vals[:-1])
        final_vals[-1] = forced_last
        if sequential_sum(final_vals) != 100.0:
            raise RuntimeError("Unable to force float-sum to exactly 100.0")

    # Per-set nudge limit check (<= 0.002)
    for a, b in zip(final_vals, base_vals):
        if abs(a - b) > 0.002 + 1e-9:
            raise RuntimeError(f"Nudge exceeded: base={b} final={a} diff={a-b}")

    # Build strings:
    # - non-last always as 3 decimals
    # - last: use 3 decimals if it still sums to 100.0 when parsed; else use repr(float) (sparingly).
    pct_strs: List[str] = [f"{v:.3f}" for v in final_vals[:-1]]
    prefix_sum = sequential_sum([float(s) for s in pct_strs])

    cand = f"{final_vals[-1]:.3f}"
    if prefix_sum + float(cand) == 100.0:
        pct_strs.append(cand)
        final_vals[-1] = float(cand)
    else:
        pct_strs.append(repr(final_vals[-1]))

    # Final parse-sum check (the bot will parse from the strings)
    if sequential_sum([float(s) for s in pct_strs]) != 100.0:
        raise RuntimeError("String parse-sum is not exactly 100.0")

    return pct_strs, final_vals, base_vals


def build_command(set_rows: List[SetRow], pf: PassFailRow) -> Tuple[str, str]:
    """
    Returns (header_line, command_line)
    """
    # Effective weights and percentages
    effs = [sr.items * sr.weight for sr in set_rows]
    pct_strs, _final_vals, _base_vals = compute_percent_strings(effs)

    parts = []
    for sr, pct in zip(set_rows, pct_strs):
        kotoba_set = "stoch" + sr.set_name.strip().lower()
        parts.append(f"{kotoba_set}({pct}%)")

    cmd = "k!quiz " + "+".join(parts) + f" {pf.pass_threshold} mmq={pf.total_questions - pf.pass_threshold} hardcore nd dauq=1"

    acc = (pf.pass_threshold / pf.total_questions) * 100.0
    header = f"Level {pf.level}: {pf.pass_threshold} Points ({pf.pass_threshold}/{pf.total_questions}={acc:.1f}%)"
    return header, cmd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    pass_fail_path = base_dir / "quiz_pass_fail_marks.csv"
    manual_path = base_dir / "manual-input-quiz.csv"
    out_path = base_dir / "15_quiz_commands.txt"

    manual_by_level = read_manual_input(manual_path)
    pf_by_level = read_pass_fail(pass_fail_path)

    levels = sorted(pf_by_level.keys())
    if levels != list(range(1, 16)):
        raise ValueError(f"Expected levels 1..15 in quiz_pass_fail_marks.csv, got: {levels}")

    blocks: List[str] = []
    for lvl in range(1, 16):
        if lvl not in manual_by_level:
            raise ValueError(f"Missing level {lvl} in manual-input-quiz.csv")
        header, cmd = build_command(manual_by_level[lvl], pf_by_level[lvl])
        blocks.append(header + "\n```\n" + cmd + "\n```")

    out_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
