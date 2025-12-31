#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weibull_calibrate_from_immersion.py

Goal
----
Calibrate a Weibull-style mastery curve F(N/b) to match your *immersion story* as closely as possible,
where the "ground truth" Pr(T <= N) is computed EXACTLY (fully discrete-time, no Poisson) for m<=4 components
and k=3 sightings-to-learn, using the same "per token => presence/absence mask" semantics as your pipeline.

We fit a *single global* Weibull shape parameter alpha (and optionally a free scale factor beta) to minimize error
between:
  - p_true(N)   := Pr(T <= N) computed from the exact multinomial mask model
  - p_pred(N)   := 1 - exp(-(beta*N/b)^alpha)

By default, we use the "mean-matched Weibull":
  beta(alpha) = Gamma(1 + 1/alpha)
so that E[T] = b exactly under the Weibull approximation (consistent with b being an expected hitting time).

Slip/guess
----------
Your quiz adds noise:
  p_correct = g + (1 - s - g) * p_mastered
with configurable s (slip) and g (guess). This is a linear transform and does not change the *best alpha*
if you use MSE on probabilities, but it matters if you choose cross-entropy as the calibration loss.
This script supports both loss types.

Inputs (put next to this script)
--------------------------------
- kanjiwords_jmdictslice.csv
    columns: word, reading, freq, rawrank
  (rawrank is used as your proxy for frequency; token weight is 1/rawrank)

- JmdictFurigana_For_ChatGPT.txt
    lines: surface|reading|seg;seg;...
    seg:  idx:kana   or   a-b:kana
  Components are keys like "明[めい]" derived from surface substrings + kana chunk.
  If mapping is missing/empty for (word,reading), we fallback to one component key:  f"{word}[{reading}]".

Outputs
-------
- weibull_calibration_params.json
- prints a human-readable summary including RMSE/MAE on the sampled calibration sets.

Notes
-----
- This script ONLY calibrates parameters (alpha, beta). It does NOT fit learner abilities N nor unique_readings.
  After calibration, to infer learner N from quiz outcomes y_j:
    p_mastered_j(N) = 1 - exp(-(beta*N/b_j)^alpha)
    p_correct_j(N)  = g + (1 - s - g)*p_mastered_j(N)
  then minimize total negative log-likelihood over N (1D search).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import math
import csv
import random
import bisect
import json
import time
from pathlib import Path


# -----------------------
# Config
# -----------------------
KANJIWORDS_CSV = "kanjiwords_jmdictslice.csv"
FURIGANA_TXT   = "JmdictFurigana_For_ChatGPT.txt"

K_SIGHTINGS = 3
MAX_COMPONENTS_FOR_CALIB = 4  # you quiz <=4
USE_ALL_WORDS = True         # If True, calibrate using ALL words (m<=MAX_COMPONENTS_FOR_CALIB)
WORD_WEIGHT_MODE = "uniform_words"  # 'uniform_words' counts each word equally; 'token_prob' weights by p_word


# Evaluate ground-truth CDF at N ≈ ratio * b  (b = expected time for that set)
RATIO_GRID = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]

# Noise model (configurable)
SLIP_S  = 0.05
GUESS_G = 0.05

# Loss for calibration: "mse" or "xent" (cross-entropy on the noisy correctness probs)
LOSS_TYPE = "mse"

# Alpha search range
ALPHA_MIN = 0.30
ALPHA_MAX = 6.00
ALPHA_STEP_COARSE = 0.05
ALPHA_STEP_FINE   = 0.005
FINE_WINDOW = 0.20  # refine +/- this around the best coarse alpha

# Optional: allow free scale beta (instead of mean-matched beta=Gamma(1+1/alpha))
# If True: fit (alpha, beta) by grid over alpha and closed-form beta on log scale is not available,
# so we do a simple 1D search for beta per alpha.
ALLOW_FREE_BETA = False
BETA_MIN = 0.10
BETA_MAX = 10.0
BETA_STEPS = 60  # log-spaced beta grid per alpha if ALLOW_FREE_BETA=True

OUTPUT_JSON = "weibull_calibration_params.json"


# -----------------------
# Furigana parsing -> component keys (matches your pipeline)
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
            if not seg:
                continue
            if ":" not in seg:
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
    Convert segments (a,b,kana) to component keys:  surface[a:b+1] + "[" + kana + "]"
    Dedup within the word (presence/absence per token).
    """
    comps: List[str] = []
    seen = set()
    for a, b, kana in segs:
        if a < 0 or b < a:
            continue
        if b >= len(surface):
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
# Exact p_mask for a component set, matching your pipeline:
#   - each token contributes at most once per component (presence/absence)
# -----------------------
def mask_probs_for_component_set(comp_ids: Tuple[int, ...], inverted_index: List[List[int]], p_word: List[float]) -> List[float]:
    """
    For fixed comp_ids of length m, compute p_mask over masks 0..2^m-1
    where mask bit i is 1 if component comp_ids[i] appears in the token.
    """
    m = len(comp_ids)
    if m == 0:
        return [1.0]
    word_mask: Dict[int, int] = {}
    for i, cid in enumerate(comp_ids):
        bit = 1 << i
        for w in inverted_index[cid]:
            word_mask[w] = word_mask.get(w, 0) | bit

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


# -----------------------
# Exact expected time (mean hitting time) to learn all m components with threshold k
# (same DP style as your code; m<=8 is fine, we use m<=4 here)
# -----------------------
def expected_time_to_learn_all(p_mask: List[float], m: int, k: int) -> float:
    """
    DP over states in base (k+1). state = (c0..c_{m-1}) with each ci in [0..k].
    goal = all ci==k.

    E[goal]=0
    For other states s:
      E[s] = 1 + sum_{mask} p_mask[mask] * E[next(s,mask)]
    Solve as:
      E[s] = (1 + sum_{mask!=0} p * E[next]) / (1 - p_mask[0])  BUT only if mask=0 means "no progress".
    Here we do the general rearrangement by isolating self-loop probability directly.
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
        for i in range(m):
            if mask & (1 << i):
                bits.append(i)
        mask_bits.append(bits)

    # Iterate states backwards (monotone / upper triangular structure)
    # This converges in one pass because next-state always >= current in this encoding
    for s in range(goal - 1, -1, -1):
        # decode counts
        counts = [0] * m
        tmp = s
        for i in range(m):
            counts[i] = tmp % base
            tmp //= base

        denom = 1.0
        numer = 1.0  # the "+1" turn cost
        # accumulate expected next
        for mask in range(1 << m):
            p = p_mask[mask]
            if p <= 0.0:
                continue
            # compute next state index
            nxt_counts = counts[:]  # m<=4 so cheap
            for i in mask_bits[mask]:
                if nxt_counts[i] < k:
                    nxt_counts[i] += 1
            # encode
            nxt = 0
            for i in range(m):
                nxt += nxt_counts[i] * base_pows[i]

            if nxt == s:
                denom -= p
            else:
                numer += p * E[nxt]

        if denom <= 0.0:
            # If denom==0, the chain cannot leave this state under p_mask; mean is infinite.
            # In practice you should not hit this for learnable sets.
            E[s] = float("inf")
        else:
            E[s] = numer / denom

    return E[0]


# -----------------------
# EXACT Pr(T <= N) for m<=4, k=3, presence/absence masks
# via inclusion-exclusion on events Xi <= k-1, using tiny constrained multinomial enumeration.
#
# Key trick: because k=3, "not learned" means count <= 2, which forces the number of nonzero-mask trials
# to be small, so the enumeration is tiny (r=4 has only 1039 feasible assignments).
# -----------------------
@dataclass(frozen=True)
class Assignment:
    # for projected masks 1..(2^r-1), store only (mask,count) with count>0
    pairs: Tuple[Tuple[int, int], ...]
    s_total: int
    log_denom: float  # sum log(count!)


_ASSIGNMENTS_CACHE: Dict[Tuple[int, int], List[Assignment]] = {}  # (r, max_low) -> list


def _generate_assignments(r: int, max_low: int) -> List[Assignment]:
    """
    Enumerate all count assignments over nonzero masks u in [1..2^r-1]
    such that for each component i, sum_{u: bit i=1} n_u <= max_low.

    Returns a list including the empty assignment (all counts 0).
    """
    key = (r, max_low)
    if key in _ASSIGNMENTS_CACHE:
        return _ASSIGNMENTS_CACHE[key]

    masks = list(range(1, 1 << r))
    out: List[Assignment] = []

    # recursion over masks with remaining budgets per component
    def rec(j: int, rem: List[int], pairs_acc: List[Tuple[int, int]], s_acc: int, logden_acc: float) -> None:
        if j == len(masks):
            out.append(Assignment(pairs=tuple(pairs_acc), s_total=s_acc, log_denom=logden_acc))
            return
        u = masks[j]
        involved = [i for i in range(r) if (u >> i) & 1]
        if involved:
            cap = min(rem[i] for i in involved)
        else:
            cap = 0
        for x in range(cap + 1):
            if x == 0:
                rec(j + 1, rem, pairs_acc, s_acc, logden_acc)
            else:
                rem2 = rem[:]  # small r
                for i in involved:
                    rem2[i] -= x
                pairs_acc.append((u, x))
                rec(j + 1, rem2, pairs_acc, s_acc + x, logden_acc + math.lgamma(x + 1.0))
                pairs_acc.pop()

    rec(0, [max_low] * r, [], 0, 0.0)
    _ASSIGNMENTS_CACHE[key] = out
    return out


def _prob_all_counts_le(N: int, q: List[float], r: int, max_low: int) -> float:
    """
    Exact probability that for r components, each count <= max_low after N iid trials,
    where each trial produces a projected mask u in [0..2^r-1] with probability q[u].

    Uses precomputed feasible assignments of nonzero mask counts.
    """
    if r == 0:
        return 1.0
    if N < 0:
        return 0.0

    q0 = q[0]
    # quick exits
    if N == 0:
        return 1.0  # all counts are 0 <= max_low
    if q0 >= 1.0 - 1e-18:
        # almost always no component appears
        return 1.0 if max_low >= 0 else 0.0

    assignments = _generate_assignments(r, max_low)

    # compute in log domain with log-sum-exp
    log_terms: List[float] = []

    # precompute logs for masks that appear
    logq = [float("-inf")] * (1 << r)
    for u in range(1 << r):
        if q[u] > 0.0:
            logq[u] = math.log(q[u])

    logq0 = logq[0]  # may be -inf if q0=0

    for a in assignments:
        s = a.s_total
        if s > N:
            continue
        n0 = N - s
        if n0 > 0 and not math.isfinite(logq0):
            # q0=0 but need n0>0 -> impossible
            continue

        # falling factorial: N*(N-1)*...*(N-s+1)
        # s<=2r<=8 so stable to compute as sum log(N - t)
        log_fall = 0.0
        for t in range(s):
            log_fall += math.log(N - t)

        # multiply probabilities
        log_p = log_fall - a.log_denom
        if n0 > 0:
            log_p += n0 * logq0
        # for each nonzero mask count
        ok = True
        for u, cnt in a.pairs:
            if not math.isfinite(logq[u]):
                ok = False
                break
            log_p += cnt * logq[u]
        if not ok:
            continue
        log_terms.append(log_p)

    if not log_terms:
        return 0.0

    m = max(log_terms)
    ssum = 0.0
    for lt in log_terms:
        ssum += math.exp(lt - m)
    return math.exp(m) * ssum


def prob_T_leq_N_from_pmask(N: int, p_mask: List[float], m: int, k: int) -> float:
    """
    Exact Pr(T <= N) for learning all m components with threshold k,
    under iid per-turn masks with probabilities p_mask[0..2^m-1].

    Because counts are monotone, "absorbed by N" iff each component count >= k at time N.

    We compute P(all counts >= k) using inclusion-exclusion on events (count_i <= k-1).
    For k=3, the constraint is <=2 (tiny enumeration).
    """
    if m == 0:
        return 1.0
    if N < k:
        return 0.0

    max_low = k - 1
    total = 0.0

    # loop over subsets A of components: A_mask in [0..2^m-1]
    for A_mask in range(1 << m):
        r = A_mask.bit_count()
        sign = -1.0 if (r % 2 == 1) else 1.0
        if r == 0:
            total += sign * 1.0
            continue

        idxs = [i for i in range(m) if (A_mask >> i) & 1]
        # project p_mask onto subset idxs -> q over 2^r masks
        q = [0.0] * (1 << r)
        for full in range(1 << m):
            proj = 0
            for pos, orig_i in enumerate(idxs):
                if (full >> orig_i) & 1:
                    proj |= (1 << pos)
            q[proj] += p_mask[full]

        p_le = _prob_all_counts_le(N, q, r, max_low)
        total += sign * p_le

    # numerical clamp
    if total < 0.0 and total > -1e-10:
        total = 0.0
    if total > 1.0 and total < 1.0 + 1e-10:
        total = 1.0
    return max(0.0, min(1.0, total))


# -----------------------
# Weibull approximation (mean-matched by default)
# -----------------------
def weibull_beta_meanmatched(alpha: float) -> float:
    # beta = Gamma(1 + 1/alpha) so that E[T]=b when scale is expressed via b
    return math.gamma(1.0 + 1.0 / alpha)

def weibull_mastery_prob(N: float, b: float, alpha: float, beta: float) -> float:
    if b <= 0.0:
        return 0.0
    x = beta * (N / b)
    if x <= 0.0:
        return 0.0
    # 1 - exp(-(x^alpha))
    z = -(x ** alpha)
    # for large negative z, exp(z) underflows to 0 -> mastery ~ 1
    return 1.0 - math.exp(z)

def noisy_correct_prob(p_mastered: float, s: float, g: float) -> float:
    # p(correct) = g + (1 - s - g) * p_mastered
    p = g + (1.0 - s - g) * p_mastered
    # clamp to avoid log(0)
    return max(1e-15, min(1.0 - 1e-15, p))

def loss_mse(p_true: float, p_pred: float) -> float:
    d = p_true - p_pred
    return d * d

def loss_xent(p_true_correct: float, p_pred_correct: float) -> float:
    # cross-entropy between Bernoulli(p_true_correct) and Bernoulli(p_pred_correct)
    return -(p_true_correct * math.log(p_pred_correct) + (1.0 - p_true_correct) * math.log(1.0 - p_pred_correct))


# -----------------------
# Data loading and component index building
# -----------------------
def load_furigana_map(path: str) -> Dict[Tuple[str, str], List[Tuple[int, int, str]]]:
    mp: Dict[Tuple[str, str], List[Tuple[int, int, str]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            surf, read, segs = parse_furigana_line(line)
            if surf and read:
                mp[(surf, read)] = segs
    return mp

def load_kanjiwords(path: str) -> Tuple[List[str], List[str], List[int]]:
    words: List[str] = []
    reads: List[str] = []
    rawranks: List[int] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            w = row["word"]
            r = row["reading"]
            rr = int(row["rawrank"])
            words.append(w)
            reads.append(r)
            rawranks.append(rr)
    return words, reads, rawranks

def build_components(words: List[str], reads: List[str], furimap: Dict[Tuple[str, str], List[Tuple[int,int,str]]]) -> Tuple[List[Tuple[int,...]], List[List[int]], Dict[str,int]]:
    """
    Returns:
      - word_comp_ids: list of component-id tuples per word
      - inverted_index: comp_id -> list of word indices containing it
      - component_to_id: mapping
    """
    component_to_id: Dict[str, int] = {}
    inverted_index: List[List[int]] = []
    word_comp_ids: List[Tuple[int, ...]] = []

    missing = 0

    for wi, (w, r) in enumerate(zip(words, reads)):
        segs = furimap.get((w, r))
        if segs:
            comp_keys = segments_to_component_keys(w, segs)
        else:
            comp_keys = []

        if not comp_keys:
            # fallback: treat whole word-reading as one component key
            missing += 1
            comp_keys = [f"{w}[{r}]"]

        ids = []
        for key in comp_keys:
            cid = component_to_id.get(key)
            if cid is None:
                cid = len(component_to_id)
                component_to_id[key] = cid
                inverted_index.append([])
            ids.append(cid)

        ids = sorted(set(ids))
        word_comp_ids.append(tuple(ids))
        for cid in ids:
            inverted_index[cid].append(wi)

    print(f"Total words: {len(words)}")
    print(f"Unique component keys: {len(component_to_id)}")
    print(f"Words missing furigana mapping (fallback used): {missing}")
    return word_comp_ids, inverted_index, component_to_id


def make_p_word(rawranks: List[int]) -> List[float]:
    weights = [1.0 / rr for rr in rawranks]
    s = sum(weights)
    return [w / s for w in weights]

def build_all_sets_with_weights(
    word_comp_ids: List[Tuple[int, ...]],
    p_word: List[float],
    max_m: int,
    weight_mode: str
) -> Dict[Tuple[int, ...], float]:
    """
    Build a mapping: comp_set -> weight, over ALL words whose component-set size is within [1..max_m].

    This satisfies "run for the whole file for every word" without recomputing ground-truth repeatedly:
      - we compute truth once per UNIQUE set
      - we weight that set by the number of words (uniform_words) or by token probability mass (token_prob)

    weight_mode:
      - 'uniform_words': weight = count of words having that set
      - 'token_prob'   : weight = sum of p_word over words having that set (immersion-weighted)
    """
    weights: Dict[Tuple[int, ...], float] = {}

    if weight_mode not in ("uniform_words", "token_prob"):
        raise ValueError("WORD_WEIGHT_MODE must be 'uniform_words' or 'token_prob'")

    for wi, s in enumerate(word_comp_ids):
        m = len(s)
        if 1 <= m <= max_m:
            if weight_mode == "uniform_words":
                weights[s] = weights.get(s, 0.0) + 1.0
            else:
                weights[s] = weights.get(s, 0.0) + p_word[wi]

    return weights


# -----------------------
# Calibration
# -----------------------
@dataclass
class SetStats:
    comp_ids: Tuple[int, ...]
    m: int
    b: float
    p_mask: List[float]
    N_list: List[int]
    p_true_list: List[float]
    weight: float


def precompute_truth_for_sets(
    set_weights: Dict[Tuple[int, ...], float],
    inverted_index: List[List[int]],
    p_word: List[float],
    ratio_grid: List[float],
    k: int
) -> List[SetStats]:
    out: List[SetStats] = []
    t0 = time.time()

    items = list(set_weights.items())
    for idx, (comp_ids, weight) in enumerate(items, 1):
        m = len(comp_ids)
        p_mask = mask_probs_for_component_set(comp_ids, inverted_index, p_word)
        b = expected_time_to_learn_all(p_mask, m, k)
        if not math.isfinite(b) or b <= 0.0:
            continue

        N_list = [max(0, int(round(r * b))) for r in ratio_grid]
        p_true_list = []
        for N in N_list:
            p_true = prob_T_leq_N_from_pmask(N, p_mask, m, k)
            p_true_list.append(p_true)

        out.append(SetStats(comp_ids=comp_ids, m=m, b=b, p_mask=p_mask, N_list=N_list, p_true_list=p_true_list, weight=weight))

        if idx % 25 == 0:
            dt = time.time() - t0
            print(f"Precomputed truth for {idx}/{len(items)} unique sets in {dt:.1f}s")

    return out


def evaluate_loss(stats: List[SetStats], alpha: float, beta: float, s: float, g: float, loss_type: str) -> float:
    tot = 0.0
    cnt = 0.0
    for st in stats:
        b = st.b
        for N, p_true in zip(st.N_list, st.p_true_list):
            p_pred = weibull_mastery_prob(N, b, alpha, beta)
            w = st.weight
            if loss_type == "mse":
                tot += w * loss_mse(p_true, p_pred)
            elif loss_type == "xent":
                pt = noisy_correct_prob(p_true, s, g)
                pp = noisy_correct_prob(p_pred, s, g)
                tot += w * loss_xent(pt, pp)
            else:
                raise ValueError("LOSS_TYPE must be 'mse' or 'xent'")
            cnt += w
    return tot / max(1e-12, cnt)


def grid_search_alpha(stats: List[SetStats], s: float, g: float, loss_type: str) -> Tuple[float, float, float]:
    """
    Returns (best_alpha, best_beta, best_loss).
    If ALLOW_FREE_BETA=False: beta is mean-matched beta(alpha)=Gamma(1+1/alpha).
    If ALLOW_FREE_BETA=True : beta is chosen by log-spaced grid per alpha.
    """
    def beta_candidates():
        # log-spaced
        if BETA_STEPS <= 1:
            return [1.0]
        lo = math.log(BETA_MIN)
        hi = math.log(BETA_MAX)
        return [math.exp(lo + (hi - lo) * i / (BETA_STEPS - 1)) for i in range(BETA_STEPS)]

    best_alpha = None
    best_beta = None
    best_loss = float("inf")

    a = ALPHA_MIN
    while a <= ALPHA_MAX + 1e-12:
        if not ALLOW_FREE_BETA:
            beta = weibull_beta_meanmatched(a)
            L = evaluate_loss(stats, a, beta, s, g, loss_type)
            if L < best_loss:
                best_loss = L
                best_alpha = a
                best_beta = beta
        else:
            # pick best beta on a grid
            local_best = float("inf")
            local_beta = None
            for beta in beta_candidates():
                L = evaluate_loss(stats, a, beta, s, g, loss_type)
                if L < local_best:
                    local_best = L
                    local_beta = beta
            if local_best < best_loss:
                best_loss = local_best
                best_alpha = a
                best_beta = local_beta
        a += ALPHA_STEP_COARSE

    # refine around the best alpha
    a0 = best_alpha
    lo = max(ALPHA_MIN, a0 - FINE_WINDOW)
    hi = min(ALPHA_MAX, a0 + FINE_WINDOW)

    a = lo
    while a <= hi + 1e-12:
        if not ALLOW_FREE_BETA:
            beta = weibull_beta_meanmatched(a)
            L = evaluate_loss(stats, a, beta, s, g, loss_type)
            if L < best_loss:
                best_loss = L
                best_alpha = a
                best_beta = beta
        else:
            local_best = float("inf")
            local_beta = None
            for beta in beta_candidates():
                L = evaluate_loss(stats, a, beta, s, g, loss_type)
                if L < local_best:
                    local_best = L
                    local_beta = beta
            if local_best < best_loss:
                best_loss = local_best
                best_alpha = a
                best_beta = local_beta
        a += ALPHA_STEP_FINE

    return float(best_alpha), float(best_beta), float(best_loss)


def summarize_fit(stats: List[SetStats], alpha: float, beta: float) -> Dict[str, float]:
    """
    Report a few intuitive shape numbers: predicted ratio Nq/b at q=0.5 and q=0.9,
    and empirical RMSE/MAE over the sampled points.
    """
    # Weibull quantile ratio: Nq/b = ( (-ln(1-q))^(1/alpha) ) / beta
    def rq(q: float) -> float:
        return ((-math.log(1.0 - q)) ** (1.0 / alpha)) / beta

    # compute error metrics on mastery (not noisy correctness)
    abs_sum = 0.0
    sq_sum = 0.0
    w_sum = 0.0
    for st in stats:
        w = st.weight
        for N, p_true in zip(st.N_list, st.p_true_list):
            p_pred = weibull_mastery_prob(N, st.b, alpha, beta)
            e = p_pred - p_true
            abs_sum += w * abs(e)
            sq_sum += w * (e * e)
            w_sum += w

    mae = abs_sum / max(1e-12, w_sum)
    rmse = math.sqrt(sq_sum / max(1e-12, w_sum))

    return {
        "alpha": alpha,
        "beta": beta,
        "rq_0.5": rq(0.5),
        "rq_0.9": rq(0.9),
        "mae_mastery": mae,
        "rmse_mastery": rmse,
    }


def main() -> None:
    t0 = time.time()
    for p in [KANJIWORDS_CSV, FURIGANA_TXT]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    print("Loading data...")
    furimap = load_furigana_map(FURIGANA_TXT)
    words, reads, rawranks = load_kanjiwords(KANJIWORDS_CSV)

    print("Building component index...")
    word_comp_ids, inverted_index, comp_to_id = build_components(words, reads, furimap)

    print("Building token distribution p_word from 1/rawrank...")
    p_word = make_p_word(rawranks)

    print(f"Building calibration set weights from ALL words (m<= {MAX_COMPONENTS_FOR_CALIB})...")
    set_weights = build_all_sets_with_weights(word_comp_ids, p_word, MAX_COMPONENTS_FOR_CALIB, WORD_WEIGHT_MODE)
    print(f"Unique sets to evaluate: {len(set_weights)} (weighted by mode={WORD_WEIGHT_MODE})")

    print("Computing ground-truth Pr(T<=N) for all unique sets...")
    stats = precompute_truth_for_sets(set_weights, inverted_index, p_word, RATIO_GRID, K_SIGHTINGS)
    print(f"Usable sets after filtering (finite mean): {len(stats)}")

    print("Searching best alpha (and beta) ...")
    alpha, beta, loss = grid_search_alpha(stats, SLIP_S, GUESS_G, LOSS_TYPE)

    summary = summarize_fit(stats, alpha, beta)
    summary["loss_type"] = LOSS_TYPE
    summary["loss_value"] = loss
    summary["k_sightings"] = K_SIGHTINGS
    summary["max_components_calib"] = MAX_COMPONENTS_FOR_CALIB
    summary["ratio_grid"] = RATIO_GRID
    summary["s_slip"] = SLIP_S
    summary["g_guess"] = GUESS_G
    summary["allow_free_beta"] = ALLOW_FREE_BETA
    summary["usable_unique_sets"] = len(stats)
    summary["unique_sets_total"] = len(set_weights)
    summary["word_weight_mode"] = WORD_WEIGHT_MODE
    summary["time_seconds"] = time.time() - t0

    print("\n=== Calibration result ===")
    print(f"alpha = {alpha:.6f}")
    print(f"beta  = {beta:.6f}   ({'free' if ALLOW_FREE_BETA else 'mean-matched'})")
    print(f"loss({LOSS_TYPE}) = {loss:.10f}")
    print(f"Implied N50/b = {summary['rq_0.5']:.6f}")
    print(f"Implied N90/b = {summary['rq_0.9']:.6f}")
    print(f"MAE (mastery prob)  = {summary['mae_mastery']:.8f}")
    print(f"RMSE (mastery prob) = {summary['rmse_mastery']:.8f}")
    print(f"Time (s) = {summary['time_seconds']:.2f}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nWrote: {OUTPUT_JSON}")

    print("\nHow to use for ability inference (not implemented here):")
    print("  Given item difficulties b_j and responses y_j ∈ {0,1}:")
    print("    p_mastered_j(N) = 1 - exp(-(beta*N/b_j)^alpha)")
    print("    p_correct_j(N)  = g + (1 - s - g)*p_mastered_j(N)")
    print("  Then find N that minimizes:")
    print("    -Σ [ y_j log p_correct_j(N) + (1-y_j) log(1 - p_correct_j(N)) ]")
    print("  (1D search over N; you can do a coarse grid on log N then refine.)")


if __name__ == "__main__":
    main()
