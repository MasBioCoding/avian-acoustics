#!/usr/bin/env python3
"""Stratified spectrogram annotation -> call-density & ROC-AUC for a classifier.

This serves a local browser app that lets you annotate spectrograms positive /
negative for one target class, then estimates the classifier's call density
``P(+)`` and ROC-AUC following the "call density estimation" protocol of:

    Dumoulin et al. (2025), "The Search for Squawk: Agile Modeling in
    Bioacoustics", Section 2.2.  (claude_info/agile_modelling.pdf)

Pipeline (see README at bottom of this docstring for the why of each step):

    1. Load inference.csv (the agile_inferences run for TARGET_CLASS).
    2. Map each WAV filename -> (xcid, clip_index) -> metadata.csv row, to attach
       the recordist and lat/lon.  (``<species>_<xcid>_<clip>.wav``: the LAST two
       underscore-separated tokens are the xcid and the clip index.)
    3. (Optional) Score filter(s): prune the pool to clips scoring at or above
       a logit threshold on a *different* classifier run of the same species
       (``--filter CLASS[:THRESHOLD]``), e.g. keep only songs via the
       song-vs-call classifier when annotating a song dialect.
    4. De-duplicate to at most one clip per unique recordist, keeping that
       recordist's highest-scoring clip.
    5. (Optional) Geographic filter: draw a polygon on the map; only candidates
       whose lat/lon fall inside are kept.  Included/excluded points are plotted
       and counted as a sanity check.
    6. Stratified sample: bucket the surviving candidates into *logarithmic*
       score quantiles (bottom 50%, next 25%, next 12.5%, ...) and draw up to
       SAMPLES_PER_BUCKET per bucket with a fixed SEED.
    7. Annotate the sampled spectrograms in a grid (positive / negative).
    8. Compute call density P(+) (Beta posteriors + bootstrap CI) and the
       quantile-decomposed ROC-AUC, and render paper-ready viridis figures.

Server uses only the standard library.  ``numpy`` + ``matplotlib`` are imported
lazily, and only when you compute metrics / figures.

Paths derive from ``--data-root`` + ``--species-slug`` + ``--target-class``:
    inference    : <data_root>/agile_inferences/<species>/<target_class>/inference.csv
    metadata     : <data_root>/embeddings/<species>/metadata.csv
    spectrograms : <data_root>/spectrograms/<species>
    filter runs  : <data_root>/agile_inferences/<species>/<filter_class>/inference.csv
so ``--target-class`` alone selects the run.  Use the *full* (untruncated)
inference.csv — including the many score<0 windows — otherwise the call-density
and ROC-AUC estimates are conditional on score>0 (inflated P(+), deflated AUC).

Typical usage::

    python roc_annotate.py --target-class chirp_pclip --selftest   # headless check
    python roc_annotate.py --target-class chirp_staple --open       # launch the UI
    python roc_annotate.py --target-class south_song \
        --filter song:0 --open      # candidate pool = songs only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import mimetypes
import os
import re
import shutil
import sys
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import unquote, urlparse

# ----------------------------------------------------------------------------
# CONFIG  --  edit these, or override any of them on the command line.
# ----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

# Data root on the external drive (holds spectrograms/, agile_inferences/, ...).
DATA_ROOT = Path(os.environ.get("BIRDCLUSTER_DATA_ROOT", "/Volumes/Z Slim/zslim_birdcluster"))

# The "entry name": the class to score.  It is also the agile_inferences run
# folder, so this single knob selects which inference.csv to load.
SPECIES_SLUG = "emberiza_calandra"
TARGET_CLASS = "chirp_pclip"
INFERENCE_NAME = None  # agile_inferences run folder; defaults to TARGET_CLASS

# Paths.  Leave these None to derive everything from DATA_ROOT / SPECIES_SLUG /
# TARGET_CLASS (set any of them to override):
#   inference    : <data_root>/agile_inferences/<species>/<inference_name>/inference.csv
#   metadata     : <data_root>/embeddings/<species>/metadata.csv
#   spectrograms : <data_root>/spectrograms/<species>
#   output       : <data_root>/eval_roc/<species>
INFERENCE_CSV = None
METADATA_CSV = None
SPECTROGRAM_DIR = None
OUTPUT_DIR = None

# CSV column names.
SCORE_COLUMN = "logits"
FILENAME_COLUMN = "filename"
LABEL_COLUMN = "label"
# Optional: keep only rows whose LABEL_COLUMN equals this value.  Off by default
# because the agile_inferences folder already scopes the file to one class.
LABEL_FILTER = None
# Optional: prune the candidate pool with *other* classifiers from the same
# species' agile_inferences folder.  Each entry is "CLASS[:THRESHOLD]" (default
# threshold 0): a candidate survives only if its clip scores at or above the
# threshold on that run.  E.g. ["song_vs_call:0"] keeps songs and drops calls
# while annotating a song dialect.  CLI ``--filter`` overrides this list.
SCORE_FILTERS: list[str] = []

# Sampling protocol (Section 2.2).
DEDUP_BY_RECORDIST = True
NUM_QUANTILE_BUCKETS = 6      # bottom 50%, 25%, 12.5%, 6.25%, 3.125%, top 3.125%
SAMPLES_PER_BUCKET = 40       # "K" examples validated per bucket
SEED = 13                     # reproducible sampling

# Estimation.
BETA_PRIOR_ALPHA = 1.0        # Beta(1,1) == uniform prior on P(+|bucket)
BETA_PRIOR_BETA = 1.0
BOOTSTRAP_SAMPLES = 10000     # bootstrap draws for the P(+) / AUC credible band

# UI / server.
DEBUG_MODE = False            # show logit + recordist + lat/lon on each card
HOST = "127.0.0.1"
PORT = 8791
AUTO_OPEN = False

# Default map view (only matters before any points load).
MAP_CENTER_LAT = 52.0
MAP_CENTER_LON = 10.0
MAP_ZOOM = 4

ANNOTATION_VALUES = {"positive", "negative"}
ANNOTATION_FIELDNAMES = [
    "row_key",
    "annotation",
    "annotated_at",
    "target_class",
    "filename",
    "png_filename",
    "xcid",
    "clip_index",
    "recordist",
    "lat",
    "lon",
    "score",
    "bucket_index",
    "bucket_population",
    "bucket_sampled",
    "weight",
    "rank_pct",
    "inference_csv",
]
WAV_STEM_RE = re.compile(r"^(?P<rest>.+)_(?P<xcid>\d+)_(?P<clip>\d+)$")


# ----------------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------------
@dataclass
class Candidate:
    """One inference row joined with its xeno-canto metadata."""

    row_key: str
    filename: str
    png_filename: str
    xcid: str
    clip_index: str
    recordist: str
    lat: float | None
    lon: float | None
    score: float
    line_number: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class SampledItem:
    """A candidate selected for annotation, with its stratum bookkeeping."""

    candidate: Candidate
    bucket_index: int
    bucket_population: int
    bucket_sampled: int
    weight: float
    rank_pct: float
    image_exists: bool = False


@dataclass
class JoinStats:
    """Diagnostics gathered while loading + joining the inference CSV."""

    total_rows: int = 0
    label_matched: int = 0
    missing_filename: int = 0
    missing_score: int = 0
    unparsed_filename: int = 0
    metadata_missing: int = 0
    joined: int = 0
    after_filters: int = 0
    recordists: int = 0
    after_dedup: int = 0


@dataclass
class ScoreFilter:
    """Candidate-pool filter: a second classifier's logits + a threshold.

    Candidates survive only if their clip scores at or above ``threshold`` on
    this run (e.g. a song-vs-call classifier with threshold 0 keeps songs).
    Filters run before recordist dedup, so dedup picks each recordist's best
    clip *among those that pass*.
    """

    name: str
    threshold: float
    inference_csv: Path
    scores: dict[str, float] = field(default_factory=dict, repr=False)
    # Funnel counters, filled while filtering: a candidate stops at the first
    # filter that rejects it, so later filters never see it.
    excluded_below: int = 0
    missing_score: int = 0


# ----------------------------------------------------------------------------
# Loading + joining
# ----------------------------------------------------------------------------
def parse_xcid_clip(filename: str) -> tuple[str, str] | None:
    """Return ``(xcid, clip_index)`` parsed from a WAV filename.

    The convention is ``<species...>_<xcid>_<clip>.wav``; we take the last two
    underscore tokens.  The clip index is normalised (``"01"`` -> ``"1"``) to
    match the integer ``clip_index`` column in metadata.csv.
    """

    stem = Path(filename).stem
    match = WAV_STEM_RE.match(stem)
    if not match:
        return None
    xcid = match.group("xcid").lstrip("0") or "0"
    clip = match.group("clip").lstrip("0") or "0"
    return xcid, clip


def load_metadata_index(metadata_csv: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Index metadata.csv by ``(xcid, clip_index)`` with both normalised to int."""

    index: dict[tuple[str, str], dict[str, str]] = {}
    with metadata_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, {"xcid", "clip_index", "recordist"}, metadata_csv)
        for row in reader:
            xcid = str(row.get("xcid", "")).strip().lstrip("0") or "0"
            clip_raw = str(row.get("clip_index", "")).strip()
            try:
                clip = str(int(float(clip_raw)))
            except ValueError:
                continue
            index[(xcid, clip)] = row
    return index


def to_float(value: str | None) -> float | None:
    """Parse a float or return ``None``."""

    if value is None:
        return None
    try:
        parsed = float(str(value).strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def row_key_for(xcid: str, clip: str, target_class: str) -> str:
    """Stable identity for one annotated clip+class, independent of sampling."""

    identity = "\x1f".join((xcid, clip, target_class))
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]


def png_name_from_wav(filename: str) -> str:
    """Derive the spectrogram PNG basename from a WAV filename."""

    cleaned = Path(filename).name
    return Path(cleaned).with_suffix(".png").name if cleaned else ""


def load_candidates(
    *,
    inference_csv: Path,
    metadata_index: dict[tuple[str, str], dict[str, str]],
    target_class: str,
    label_filter: str | None,
    score_column: str,
    filename_column: str,
    label_column: str,
) -> tuple[list[Candidate], JoinStats]:
    """Load inference rows and join metadata.

    ``target_class`` namespaces the annotation row keys (so labels for different
    classes never collide).  ``label_filter`` optionally restricts rows to one
    ``label_column`` value; it is usually ``None`` because the agile_inferences
    folder already scopes the file to a single class.
    """

    stats = JoinStats()
    candidates: list[Candidate] = []
    with inference_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {score_column, filename_column}
        if label_filter is not None:
            required.add(label_column)
        require_columns(reader.fieldnames, required, inference_csv)
        for line_number, row in enumerate(reader, start=2):
            stats.total_rows += 1
            if label_filter is not None and (row.get(label_column) or "").strip() != label_filter:
                continue
            stats.label_matched += 1

            filename = (row.get(filename_column) or "").strip()
            if not filename:
                stats.missing_filename += 1
                continue
            score = to_float(row.get(score_column))
            if score is None:
                stats.missing_score += 1
                continue

            parsed = parse_xcid_clip(filename)
            if parsed is None:
                stats.unparsed_filename += 1
                continue
            xcid, clip = parsed
            meta = metadata_index.get((xcid, clip))
            if meta is None:
                stats.metadata_missing += 1
                continue

            stats.joined += 1
            candidates.append(
                Candidate(
                    row_key=row_key_for(xcid, clip, target_class),
                    filename=Path(filename).name,
                    png_filename=png_name_from_wav(filename),
                    xcid=xcid,
                    clip_index=clip,
                    recordist=(meta.get("recordist") or "").strip(),
                    lat=to_float(meta.get("lat")),
                    lon=to_float(meta.get("lon")),
                    score=score,
                    line_number=line_number,
                    metadata={
                        "country": (meta.get("country") or "").strip(),
                        "location": (meta.get("location") or "").strip(),
                        "date": (meta.get("date") or "").strip(),
                        "quality": (meta.get("quality") or "").strip(),
                        "type": (meta.get("type") or "").strip(),
                    },
                )
            )
    return candidates, stats


def dedup_by_recordist(candidates: list[Candidate]) -> list[Candidate]:
    """Keep at most one clip per recordist: the highest-scoring one.

    Ties break on the lower xcid then clip index, so the result is deterministic.
    Candidates with a blank recordist are each treated as their own group.
    """

    best: dict[str, Candidate] = {}
    for index, cand in enumerate(candidates):
        key = cand.recordist or f"__unknown__{index}"
        current = best.get(key)
        if current is None or _dedup_sort_key(cand) > _dedup_sort_key(current):
            best[key] = cand
    return sorted(best.values(), key=lambda c: (-c.score, c.xcid, c.clip_index))


def _dedup_sort_key(cand: Candidate) -> tuple[float, int, int]:
    """Higher is "better" for dedup: top score, then lower ids."""

    try:
        xcid_num = -int(cand.xcid)
    except ValueError:
        xcid_num = 0
    try:
        clip_num = -int(cand.clip_index)
    except ValueError:
        clip_num = 0
    return (cand.score, xcid_num, clip_num)


# ----------------------------------------------------------------------------
# Score filters (prune the pool with a second classifier, e.g. song vs call)
# ----------------------------------------------------------------------------
def parse_filter_spec(spec: str) -> tuple[str, float]:
    """Parse ``"CLASS[:THRESHOLD]"`` into ``(class_name, threshold)``."""

    name, sep, raw = spec.partition(":")
    name = name.strip()
    if not name:
        raise SystemExit(f"Invalid --filter spec: {spec!r} (expected CLASS[:THRESHOLD])")
    if not sep or not raw.strip():
        return name, 0.0
    try:
        return name, float(raw)
    except ValueError:
        raise SystemExit(f"Invalid --filter threshold in {spec!r} (need a number)")


def load_filter_scores(
    inference_csv: Path, *, score_column: str, filename_column: str
) -> dict[str, float]:
    """Map WAV basename -> logit from one filter run's inference.csv."""

    scores: dict[str, float] = {}
    with inference_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(reader.fieldnames, {score_column, filename_column}, inference_csv)
        for row in reader:
            filename = Path((row.get(filename_column) or "").strip()).name
            score = to_float(row.get(score_column))
            if filename and score is not None:
                scores[filename] = score
    return scores


def apply_score_filters(
    candidates: list[Candidate], filters: list[ScoreFilter]
) -> list[Candidate]:
    """Keep candidates whose clip passes every filter.

    A clip passes one filter when its logit in that run is at or above the
    threshold; clips absent from a filter's inference.csv are rejected and
    counted separately (``missing_score``).
    """

    if not filters:
        return candidates
    kept: list[Candidate] = []
    for cand in candidates:
        for flt in filters:
            score = flt.scores.get(cand.filename)
            if score is None:
                flt.missing_score += 1
                break
            if score < flt.threshold:
                flt.excluded_below += 1
                break
        else:
            kept.append(cand)
    return kept


# ----------------------------------------------------------------------------
# Geographic polygon filter
# ----------------------------------------------------------------------------
def point_in_polygon(lat: float, lon: float, polygon: list[list[float]]) -> bool:
    """Ray-casting test.  ``polygon`` is a list of ``[lat, lon]`` vertices."""

    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        yi, xi = polygon[i][0], polygon[i][1]
        yj, xj = polygon[j][0], polygon[j][1]
        intersects = ((yi > lat) != (yj > lat)) and (
            lon < (xj - xi) * (lat - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def apply_geo_filter(
    candidates: list[Candidate], polygon: list[list[float]] | None
) -> tuple[list[Candidate], list[Candidate]]:
    """Split candidates into (inside, outside) the polygon.

    With no polygon, everything that has coordinates is "inside".  Candidates
    without coordinates are always excluded when a polygon is supplied.
    """

    if not polygon:
        inside = [c for c in candidates if c.lat is not None and c.lon is not None]
        outside = [c for c in candidates if c.lat is None or c.lon is None]
        return inside, outside

    inside, outside = [], []
    for cand in candidates:
        if (
            cand.lat is not None
            and cand.lon is not None
            and point_in_polygon(cand.lat, cand.lon, polygon)
        ):
            inside.append(cand)
        else:
            outside.append(cand)
    return inside, outside


# ----------------------------------------------------------------------------
# Logarithmic-quantile stratified sampling (Section 2.2)
# ----------------------------------------------------------------------------
def log_quantile_edges(n_buckets: int) -> list[float]:
    """Cumulative rank edges for log quantiles: [0, .5, .75, .875, ..., 1].

    Bucket ``k`` (0 = lowest scores) spans cumulative fraction
    ``[edges[k], edges[k+1])``.  The construction halves the remaining top
    fraction at each step, so the final two buckets are equal width and the
    edges sum to a partition of ``[0, 1]``.
    """

    if n_buckets < 1:
        raise ValueError("n_buckets must be >= 1")
    edges = [0.0]
    for k in range(1, n_buckets):
        edges.append(1.0 - 2.0 ** (-k))
    edges.append(1.0)
    return edges


def bucket_fractions(n_buckets: int) -> list[float]:
    """P(b) by construction for each bucket (lowest -> highest score)."""

    edges = log_quantile_edges(n_buckets)
    return [edges[k + 1] - edges[k] for k in range(n_buckets)]


def stratified_sample(
    *,
    candidates: list[Candidate],
    n_buckets: int,
    samples_per_bucket: int,
    seed: int,
    spectrogram_dir: Path,
) -> tuple[list[SampledItem], list[dict[str, Any]]]:
    """Bucket candidates by log score-quantile and sample K from each.

    Returns the sampled items in a seeded *random* display order (so the
    annotator never sees clips sorted by score or grouped by bucket, which would
    anchor labels on the classifier's confidence) plus a per-bucket summary.
    ``weight = population / sampled`` lets the annotated sample stand in for the
    bucket population in the empirical ROC.
    """

    import random

    rng = random.Random(seed)
    if not candidates:
        return [], []

    ordered = sorted(candidates, key=lambda c: (c.score, c.xcid, c.clip_index))
    total = len(ordered)
    edges = log_quantile_edges(n_buckets)

    buckets: list[list[Candidate]] = [[] for _ in range(n_buckets)]
    rank_pct: dict[str, float] = {}
    for rank, cand in enumerate(ordered):
        frac = rank / total
        index = n_buckets - 1
        for k in range(n_buckets):
            if edges[k] <= frac < edges[k + 1]:
                index = k
                break
        buckets[index].append(cand)
        rank_pct[cand.row_key] = frac

    items: list[SampledItem] = []
    summaries: list[dict[str, Any]] = []
    construction = bucket_fractions(n_buckets)
    for index, members in enumerate(buckets):
        population = len(members)
        if population == 0:
            summaries.append(
                {
                    "bucket_index": index,
                    "population": 0,
                    "sampled": 0,
                    "p_b": population / total,
                    "p_b_construction": construction[index],
                    "score_min": None,
                    "score_max": None,
                }
            )
            continue

        take = population if samples_per_bucket <= 0 else min(samples_per_bucket, population)
        chosen = members if take >= population else rng.sample(members, take)
        weight = population / take if take else 0.0
        for cand in chosen:
            items.append(
                SampledItem(
                    candidate=cand,
                    bucket_index=index,
                    bucket_population=population,
                    bucket_sampled=take,
                    weight=weight,
                    rank_pct=rank_pct[cand.row_key],
                    image_exists=(spectrogram_dir / cand.png_filename).is_file(),
                )
            )
        summaries.append(
            {
                "bucket_index": index,
                "population": population,
                "sampled": take,
                "p_b": population / total,
                "p_b_construction": construction[index],
                "score_min": min(c.score for c in members),
                "score_max": max(c.score for c in members),
            }
        )
    # Randomise the grid order across all buckets (seeded -> reproducible) so
    # display order carries no information about score or bucket.
    rng.shuffle(items)
    return items, summaries


# ----------------------------------------------------------------------------
# Metrics: call density P(+) and quantile-decomposed ROC-AUC (Section 2.2)
# ----------------------------------------------------------------------------
def local_auc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Rank-based AUC of positives vs negatives (ties count 0.5).

    Returns 0.5 when either class is empty (uninformative within-bucket term).
    """

    if not pos_scores or not neg_scores:
        return 0.5
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))


def compute_metrics(
    *,
    annotated_items: list[dict[str, Any]],
    bucket_summaries: list[dict[str, Any]],
    n_buckets: int,
    prior_alpha: float,
    prior_beta: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate P(+), ROC-AUC and supporting per-bucket statistics.

    ``annotated_items`` are dicts with keys ``bucket_index``, ``score`` and
    ``annotation`` (``"positive"``/``"negative"``).  Implements the call-density
    estimator and the quantile-decomposed ROC-AUC from Section 2.2.
    """

    import numpy as np

    pop = np.array([s["population"] for s in bucket_summaries], dtype=float)
    p_b = np.array([s["p_b"] for s in bucket_summaries], dtype=float)
    total_pop = float(pop.sum())
    if total_pop <= 0:
        raise ValueError("No candidate population to estimate over.")

    pos = np.zeros(n_buckets)
    neg = np.zeros(n_buckets)
    pos_scores: list[list[float]] = [[] for _ in range(n_buckets)]
    neg_scores: list[list[float]] = [[] for _ in range(n_buckets)]
    for item in annotated_items:
        b = int(item["bucket_index"])
        if not 0 <= b < n_buckets:
            continue
        if item["annotation"] == "positive":
            pos[b] += 1
            pos_scores[b].append(float(item["score"]))
        elif item["annotation"] == "negative":
            neg[b] += 1
            neg_scores[b].append(float(item["score"]))

    n_b = pos + neg
    alpha = prior_alpha + pos
    beta = prior_beta + neg
    p_pos_given_b = alpha / (alpha + beta)            # posterior mean E[P(+|b)]
    p_neg_given_b = 1.0 - p_pos_given_b

    # --- Call density P(+) = sum_b P(+|b) P(b) -------------------------------
    density_point = float(np.sum(p_pos_given_b * p_b))

    rng = np.random.default_rng(seed)
    draws = rng.beta(
        np.broadcast_to(alpha, (bootstrap_samples, n_buckets)),
        np.broadcast_to(beta, (bootstrap_samples, n_buckets)),
    )
    density_samples = draws @ p_b
    density_lo, density_hi = (float(x) for x in np.percentile(density_samples, [2.5, 97.5]))

    # --- ROC-AUC via quantile decomposition ----------------------------------
    # P(b|+) and P(c|-) by Bayes rule from expected P(+|b), P(-|b) and P(b).
    joint_pos = p_pos_given_b * p_b
    joint_neg = p_neg_given_b * p_b
    p_b_given_pos = joint_pos / (joint_pos.sum() or 1.0)
    p_c_given_neg = joint_neg / (joint_neg.sum() or 1.0)

    local = np.array(
        [local_auc(pos_scores[b], neg_scores[b]) for b in range(n_buckets)]
    )

    def auc_from(pbp: "np.ndarray", pcn: "np.ndarray", local_vals: "np.ndarray") -> float:
        # higher bucket index == higher score, so term=1 when b>c, 0 when b<c.
        higher = np.triu(np.ones((n_buckets, n_buckets)), k=1)      # b<c upper -> 0..
        cross = higher.T.copy()                                     # b>c -> 1
        np.fill_diagonal(cross, local_vals)
        return float(pbp @ cross @ pcn)

    auc_point = auc_from(p_b_given_pos, p_c_given_neg, local)

    # Bootstrap band for the AUC: resample the bucket positive-rates only
    # (local within-bucket AUCs held at their point estimates).
    boot_pos = draws                                  # (S, n_buckets) ~ Beta
    boot_neg = 1.0 - boot_pos
    jp = boot_pos * p_b
    jn = boot_neg * p_b
    pbp = jp / np.clip(jp.sum(axis=1, keepdims=True), 1e-12, None)
    pcn = jn / np.clip(jn.sum(axis=1, keepdims=True), 1e-12, None)
    higher = np.triu(np.ones((n_buckets, n_buckets)), k=1)
    cross = higher.T.copy()
    np.fill_diagonal(cross, local)
    auc_samples = np.einsum("sb,bc,sc->s", pbp, cross, pcn)
    auc_lo, auc_hi = (float(x) for x in np.percentile(auc_samples, [2.5, 97.5]))

    # --- Diagnostics: empirical (weighted + raw) AUC on the sample -----------
    weighted_auc, weighted_roc = empirical_weighted_roc(annotated_items, np, weighted=True)
    raw_auc, raw_roc = empirical_weighted_roc(annotated_items, np, weighted=False)

    per_bucket = []
    for b, summ in enumerate(bucket_summaries):
        per_bucket.append(
            {
                "bucket_index": b,
                "score_min": summ["score_min"],
                "score_max": summ["score_max"],
                "population": int(pop[b]),
                "p_b": float(p_b[b]),
                "annotated": int(n_b[b]),
                "positive": int(pos[b]),
                "negative": int(neg[b]),
                "p_pos_given_b": float(p_pos_given_b[b]),
                "p_pos_ci": [
                    float(x) for x in beta_ci(alpha[b], beta[b], np)
                ],
                "local_auc": float(local[b]),
                # Beta posterior params (ridgeline figure) + Bayes-rule
                # conditionals (kept for metrics.json).
                "alpha": float(alpha[b]),
                "beta": float(beta[b]),
                "p_b_given_pos": float(p_b_given_pos[b]),
                "p_c_given_neg": float(p_c_given_neg[b]),
            }
        )

    return {
        "n_buckets": n_buckets,
        "n_annotated": int(n_b.sum()),
        "n_positive": int(pos.sum()),
        "n_negative": int(neg.sum()),
        "density": {
            "point": density_point,
            "ci95": [density_lo, density_hi],
            "_samples": density_samples,
        },
        "roc_auc": {
            "point": auc_point,
            "ci95": [auc_lo, auc_hi],
            "weighted_empirical": weighted_auc,
            "raw_empirical": raw_auc,
            "_samples": auc_samples,
            "_weighted_roc": weighted_roc,
            "_raw_roc": raw_roc,
        },
        "per_bucket": per_bucket,
    }


def beta_ci(alpha: float, beta_param: float, np: Any, q: float = 0.95) -> tuple[float, float]:
    """Equal-tailed credible interval for Beta(alpha, beta) via sampling."""

    lo = (1 - q) / 2
    sample = np.random.default_rng(0).beta(alpha, beta_param, size=4000)
    return tuple(np.percentile(sample, [lo * 100, (1 - lo) * 100]))


def empirical_weighted_roc(
    annotated_items: list[dict[str, Any]], np: Any, *, weighted: bool
) -> tuple[float, dict[str, list[float]]]:
    """Empirical ROC over the annotated sample (optionally stratum-weighted)."""

    labelled = [it for it in annotated_items if it["annotation"] in ANNOTATION_VALUES]
    if not labelled:
        return float("nan"), {"fpr": [], "tpr": []}
    scores = np.array([float(it["score"]) for it in labelled])
    y = np.array([1.0 if it["annotation"] == "positive" else 0.0 for it in labelled])
    w = np.array([float(it.get("weight", 1.0)) if weighted else 1.0 for it in labelled])

    pos_w = float(np.sum(w * y))
    neg_w = float(np.sum(w * (1 - y)))
    if pos_w <= 0 or neg_w <= 0:
        return float("nan"), {"fpr": [], "tpr": []}

    order = np.argsort(-scores)
    tp = np.cumsum(w[order] * y[order]) / pos_w
    fp = np.cumsum(w[order] * (1 - y[order])) / neg_w
    fpr = np.concatenate([[0.0], fp])
    tpr = np.concatenate([[0.0], tp])
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    auc = float(trapezoid(tpr, fpr))
    # thresholds[i] is the classifier logit of the point added at vertex i+1
    # (i.e. the score *above* which points count positive at that vertex).
    return auc, {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": scores[order].tolist(),
    }


# ----------------------------------------------------------------------------
# Paper-ready figures (viridis)
# ----------------------------------------------------------------------------
# Shared visual language with the Bokeh figures of xc_scripts/kde_map_animate.py
# (the "Monthly sample size" panel): a viridis-dark plot panel, light-blue
# border, green bars, and that panel's font sizes.  The score-distribution
# figure adds two annotation accents (yellow positives, red negatives) and a
# yellow bucket-boundary line, taken from the map's isocline anchors.
FIG_TITLE_FONT_SIZE = 24             # MONTHLY_SAMPLE_TITLE_FONT_SIZE ("24pt")
FIG_TITLE_PAD = 13                   # raise the bold title a sliver (~7px) off the panel
FIG_AXIS_LABEL_FONT_SIZE = 20        # MONTHLY_SAMPLE_AXIS_LABEL_FONT_SIZE ("20pt")
FIG_AXIS_MAJOR_LABEL_FONT_SIZE = 16  # MONTHLY_SAMPLE_AXIS_MAJOR_LABEL_FONT_SIZE
FIG_BACKGROUND_COLOR = "#404788"     # plot-area fill (viridis dark blue)
FIG_BORDER_COLOR = "#dcecf7"         # figure fill outside the axes (light blue)
FIG_DATAPOINT_COLOR = "#73D055"      # histogram bars (viridis green)
FIG_DATAPOINT_ALPHA = 0.85
FIG_BUCKET_COLOR = "#f4d03f"         # dashed log-quantile bucket boundaries
FIG_POSITIVE_COLOR = "#f4d03f"       # positive annotation rug
FIG_NEGATIVE_COLOR = "#e53935"       # negative annotation rug


def render_figures(
    *,
    metrics: dict[str, Any],
    population_scores: list[float],
    annotated_items: list[dict[str, Any]],
    bucket_summaries: list[dict[str, Any]],
    target_class: str,
    species_slug: str,
    output_dir: Path,
) -> list[str]:
    """Render and save the paper figures.  Returns saved file paths.

    Figures: score distribution, per-bucket P(+|b), the weighted empirical ROC
    (threshold-labelled), call-density bootstrap, per-bucket Beta posteriors,
    bucket-population sanity check, and an AUC estimator comparison.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )
    cmap = plt.get_cmap("viridis")
    output_dir.mkdir(parents=True, exist_ok=True)
    n_buckets = metrics["n_buckets"]
    bucket_colors = [cmap(i / max(n_buckets - 1, 1)) for i in range(n_buckets)]
    saved: list[str] = []

    def _save(fig: Any, name: str) -> None:
        path = output_dir / f"{target_class}_{name}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(str(path))

    title_prefix = f"{species_slug} / {target_class}"

    # (1) Score distribution with bucket boundaries + annotated rug.
    # Styled to match the Bokeh "Monthly sample size" panel of
    # xc_scripts/kde_map_animate.py: viridis-dark panel, green bars, yellow/red
    # annotation accents, that panel's font sizes, plus a right-hand axis that
    # labels where the negative (0) and positive (1) rugs sit.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BORDER_COLOR)
    ax.set_facecolor(FIG_BACKGROUND_COLOR)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="white", alpha=0.15)
    if population_scores:
        # No stroke; instead leave a thin gap between bars (rwidth < 1) that
        # reads like the former 0.4pt white edge, revealing the dark panel.
        ax.hist(population_scores, bins=40, color=FIG_DATAPOINT_COLOR,
                alpha=FIG_DATAPOINT_ALPHA, edgecolor="none", rwidth=0.96,
                label="candidate population")
    for summ in bucket_summaries:
        if summ["score_max"] is not None:
            ax.axvline(summ["score_max"], color=FIG_BUCKET_COLOR, lw=1.2,
                       ls=":", alpha=0.9)
    pos = [it["score"] for it in annotated_items if it["annotation"] == "positive"]
    neg = [it["score"] for it in annotated_items if it["annotation"] == "negative"]
    y0, y1 = ax.get_ylim()
    y_neg, y_pos = y1 * 0.04, y1 * 0.09       # rug heights (unchanged positions)
    ax.plot(neg, [y_neg] * len(neg), "|", color=FIG_NEGATIVE_COLOR,
            ms=12, mew=1.4, label=f"negative (n={len(neg)})")
    ax.plot(pos, [y_pos] * len(pos), "|", color=FIG_POSITIVE_COLOR,
            ms=12, mew=1.4, label=f"positive (n={len(pos)})")
    ax.set_ylim(y0, y1)
    ax.set_xlabel("Logit score", fontsize=FIG_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("count", fontsize=FIG_AXIS_LABEL_FONT_SIZE)
    ax.set_title("Score distributions", fontsize=FIG_TITLE_FONT_SIZE,
                 fontweight="bold", pad=FIG_TITLE_PAD)
    ax.tick_params(axis="both", labelsize=FIG_AXIS_MAJOR_LABEL_FONT_SIZE)

    # Right-hand axis: annotation labels (negative = 0, positive = 1) aligned to
    # the existing rug heights, without moving the rugs themselves.
    ax2 = ax.twinx()
    ax2.set_ylim(y0, y1)
    ax2.set_yticks([y_neg, y_pos])
    ax2.set_yticklabels(["0", "1"])
    ax2.set_ylabel("annotation", fontsize=FIG_AXIS_LABEL_FONT_SIZE)
    ax2.tick_params(axis="y", labelsize=FIG_AXIS_MAJOR_LABEL_FONT_SIZE)
    ax2.spines["right"].set_visible(True)
    ax2.grid(False)

    leg = ax.legend(loc="upper right", fontsize=11)
    for text in leg.get_texts():
        text.set_color("white")
    _save(fig, "score_distribution")

    # (2) Per-bucket P(+|b): Beta posterior mean + CI.
    # Same figure size, palette and fonts as the score-distribution figure
    # (dark panel, green bars, light-blue accents).
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fig.patch.set_facecolor(FIG_BORDER_COLOR)
    ax.set_facecolor(FIG_BACKGROUND_COLOR)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="white", alpha=0.15)
    xs = list(range(n_buckets))
    means = [pb["p_pos_given_b"] for pb in metrics["per_bucket"]]
    los = [pb["p_pos_ci"][0] for pb in metrics["per_bucket"]]
    his = [pb["p_pos_ci"][1] for pb in metrics["per_bucket"]]
    err = [[m - lo for m, lo in zip(means, los)], [hi - m for m, hi in zip(means, his)]]
    # No stroke; keep the default bar width/spacing.
    ax.bar(xs, means, color=FIG_DATAPOINT_COLOR, alpha=FIG_DATAPOINT_ALPHA,
           edgecolor="none")
    ax.errorbar(xs, means, yerr=err, fmt="none", ecolor=FIG_BORDER_COLOR,
                capsize=4, lw=1.2)
    for x, pb in zip(xs, metrics["per_bucket"]):
        ax.text(x, 1.03, f"{pb['positive']}/{pb['annotated']}",
                ha="center", va="bottom", fontsize=11, color="white")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"b{b}" for b in xs])
    ax.set_ylim(0, 1.15)
    ax.set_xlabel("bucket", fontsize=FIG_AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("P(+ | bucket)", fontsize=FIG_AXIS_LABEL_FONT_SIZE)
    ax.set_title("Positive rate per bucket", fontsize=FIG_TITLE_FONT_SIZE,
                 fontweight="bold", pad=FIG_TITLE_PAD)
    ax.tick_params(axis="both", labelsize=FIG_AXIS_MAJOR_LABEL_FONT_SIZE)
    _save(fig, "positive_rate_per_bucket")

    def _style_roc_axes(ax: Any) -> None:
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")

    auc = metrics["roc_auc"]

    # The decomposed-ROC figure was removed; drop any stale copy so the output
    # folder only holds figures the current code produces.
    (output_dir / f"{target_class}_roc_curve_decomposed.png").unlink(missing_ok=True)

    # (3) Empirical ROC: the stratum-weighted curve over the sample, with the
    # unweighted (raw-sample) curve drawn alongside for comparison.
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    roc = auc["_weighted_roc"]
    raw = auc.get("_raw_roc") or {"fpr": [], "tpr": []}
    ax.plot([0, 1], [0, 1], ls=":", color="#999999", lw=1.0)
    if raw["fpr"]:
        ax.plot(raw["fpr"], raw["tpr"], color=cmap(0.65), lw=1.8, ls="--",
                zorder=5, label=f"unweighted empirical (AUC={auc['raw_empirical']:.3f})")
    if roc["fpr"]:
        ax.plot(roc["fpr"], roc["tpr"], color=cmap(0.30), lw=2.0, zorder=6,
                label=f"weighted empirical (AUC={auc['weighted_empirical']:.3f})")
        thr = roc.get("thresholds", [])
        if thr:
            idxs = sorted({int(round(x)) for x in np.linspace(0, len(thr) - 1, 6)})
            for i in idxs:
                fx, fy = roc["fpr"][i + 1], roc["tpr"][i + 1]
                ax.plot(fx, fy, "o", color=cmap(0.30), ms=4, zorder=7)
                ax.annotate(f"{thr[i]:.1f}", (fx, fy), textcoords="offset points",
                            xytext=(4, -10), fontsize=7.5, color="#444444")
    _style_roc_axes(ax)
    ax.set_title(f"{title_prefix}: empirical ROC\n"
                 f"weighted AUC = {auc['weighted_empirical']:.3f}   "
                 f"unweighted AUC = {auc['raw_empirical']:.3f}")
    ax.legend(loc="lower right", fontsize=9)
    ax.text(0.5, -0.16, "number labels = classifier logit threshold",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            color="#666666", style="italic")
    _save(fig, "roc_curve_empirical")

    # (4) Call density P(+) bootstrap distribution.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    samples = np.asarray(metrics["density"]["_samples"])
    ax.hist(samples, bins=60, color=cmap(0.5), alpha=0.7, edgecolor="white", linewidth=0.3)
    point = metrics["density"]["point"]
    lo, hi = metrics["density"]["ci95"]
    ax.axvline(point, color=cmap(0.9), lw=2.0, label=f"P(+) = {point:.4f}")
    ax.axvspan(lo, hi, color=cmap(0.2), alpha=0.18, label=f"95% CI [{lo:.4f}, {hi:.4f}]")
    ax.set_xlabel("call density  P(+)")
    ax.set_ylabel("bootstrap count")
    ax.set_title(f"{title_prefix}: estimated call density")
    ax.legend(loc="upper right")
    _save(fig, "call_density")

    # (5) Per-bucket Beta posteriors for P(+|bucket): a ridgeline.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    grid = np.linspace(0.0, 1.0, 400)
    log_g = np.log(np.clip(grid, 1e-9, 1.0))
    log_1mg = np.log(np.clip(1.0 - grid, 1e-9, 1.0))
    yticks, ylabels = [], []
    for b, (pb, color) in enumerate(zip(metrics["per_bucket"], bucket_colors)):
        a, bt = pb["alpha"], pb["beta"]
        log_norm = math.lgamma(a + bt) - math.lgamma(a) - math.lgamma(bt)
        pdf = np.exp((a - 1.0) * log_g + (bt - 1.0) * log_1mg + log_norm)
        peak = float(pdf.max())
        if peak > 0:
            pdf = pdf / peak                      # normalise height (ridgeline)
        base = float(b)
        ax.fill_between(grid, base, base + pdf, color=color, alpha=0.7, lw=0)
        ax.plot(grid, base + pdf, color="white", lw=0.8)
        ax.plot([pb["p_pos_given_b"]] * 2, [base, base + 1.0],
                color="#222222", lw=1.0, ls="--", alpha=0.6)
        yticks.append(base)
        ylabels.append(f"b{b}  ({pb['positive']}/{pb['annotated']})")
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, n_buckets + 0.1)
    ax.set_xlabel("P(+ | bucket)")
    ax.set_title(f"{title_prefix}: per-bucket Beta posteriors  (dashed = posterior mean)")
    ax.grid(axis="y", visible=False)
    _save(fig, "beta_posteriors")

    # (6) Sanity: empirical bucket fractions vs nominal log-quantile fractions.
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = list(range(n_buckets))
    emp = [s["p_b"] for s in bucket_summaries]
    nom = [s["p_b_construction"] for s in bucket_summaries]
    width = 0.4
    ax.bar([x - width / 2 for x in xs], emp, width, color=bucket_colors,
           alpha=0.9, edgecolor="white", label="empirical  P(b) = pop / total")
    ax.bar([x + width / 2 for x in xs], nom, width, color="#bbbbbb",
           alpha=0.8, edgecolor="white", label="nominal (log-quantile)")
    for x, s in zip(xs, bucket_summaries):
        ax.text(x, max(s["p_b"], s["p_b_construction"]) + 0.012,
                f"n={s['population']}", ha="center", va="bottom",
                fontsize=8, color="#444444")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"b{b}" for b in xs])
    ax.set_xlabel("log-quantile bucket")
    ax.set_ylabel("P(bucket)")
    ax.set_title(f"{title_prefix}: bucket population vs construction")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, "bucket_population")

    # (7) Sanity: do the three AUC estimators agree?  (rare/common check)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    rows = [
        ("decomposition", auc["point"], auc["ci95"]),
        ("weighted empirical", auc["weighted_empirical"], None),
        ("raw empirical", auc["raw_empirical"], None),
    ]
    ys = list(range(len(rows) - 1, -1, -1))
    for y, (_, val, ci) in zip(ys, rows):
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        if ci is not None:
            ax.plot(ci, [y, y], color=cmap(0.30), lw=2.0, zorder=4)
            for edge in ci:
                ax.plot([edge, edge], [y - 0.09, y + 0.09], color=cmap(0.30), lw=2.0)
        ax.plot(val, y, "o", color=cmap(0.75), ms=10, zorder=6)
        ax.annotate(f"{val:.3f}", (val, y), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9)
    ax.axvline(0.5, color="#999999", ls=":", lw=1.0)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlabel("ROC-AUC")
    ax.set_title(f"{title_prefix}: AUC estimator comparison")
    _save(fig, "auc_comparison")

    return saved


# ----------------------------------------------------------------------------
# Small shared helpers
# ----------------------------------------------------------------------------
def require_columns(fieldnames: list[str] | None, required: set[str], path: Path) -> None:
    """Exit when a CSV lacks required columns."""

    missing = sorted(required - set(fieldnames or []))
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {', '.join(missing)}")


def require_readable_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def available_runs_hint(inference_csv: Path) -> str:
    """A ``\\n  available runs: ...`` hint listing sibling run folders (or '')."""

    runs_dir = inference_csv.parent.parent
    if not runs_dir.is_dir():
        return ""
    runs = sorted(p.name for p in runs_dir.iterdir() if p.is_dir())
    return f"\n  available runs: {', '.join(runs)}" if runs else ""


def json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def safe_name(name: str) -> str:
    """Sanitise a user-supplied name into a filename-safe slug (or '')."""

    cleaned = re.sub(r"[^A-Za-z0-9 _-]+", "", (name or "").strip())
    cleaned = re.sub(r"\s+", "_", cleaned).strip("._-")
    return cleaned[:64]


def load_polygons(path: Path) -> dict[str, list[list[float]]]:
    """Load named polygons (``{name: [[lat, lon], ...]}``) from disk."""

    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, list[list[float]]] = {}
    if isinstance(data, dict):
        for name, ring in data.items():
            if isinstance(ring, list) and len(ring) >= 3:
                try:
                    out[str(name)] = [[float(p[0]), float(p[1])] for p in ring]
                except (TypeError, ValueError, IndexError):
                    continue
    return out


def save_polygon(path: Path, name: str, ring: list[list[float]]) -> dict[str, list[list[float]]]:
    """Add or replace one named polygon, persisting the whole collection."""

    polygons = load_polygons(path)
    polygons[name] = ring
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(polygons, indent=2), encoding="utf-8")
    tmp.replace(path)
    return polygons


# ----------------------------------------------------------------------------
# Server state
# ----------------------------------------------------------------------------
@dataclass
class AppState:
    """Mutable server state shared across HTTP requests."""

    candidates: list[Candidate]              # deduped, before geo filter
    join_stats: JoinStats
    spectrogram_dir: Path
    output_dir: Path
    target_class: str
    species_slug: str
    inference_csv: Path
    n_buckets: int
    samples_per_bucket: int
    seed: int
    prior_alpha: float
    prior_beta: float
    bootstrap_samples: int
    debug_mode: bool
    score_filters: list[ScoreFilter] = field(default_factory=list)
    # session (set after /api/sample)
    polygon: list[list[float]] | None = None
    items: list[SampledItem] = field(default_factory=list)
    bucket_summaries: list[dict[str, Any]] = field(default_factory=list)
    filtered_scores: list[float] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    @property
    def items_by_key(self) -> dict[str, SampledItem]:
        return {it.candidate.row_key: it for it in self.items}

    @property
    def default_annotations_path(self) -> Path:
        return self.output_dir / f"{self.target_class}.annotations.csv"

    def annotations_path(self, name: str | None = None) -> Path:
        """Path for an annotation set; the unnamed set keeps the legacy filename."""

        slug = safe_name(name or "")
        if not slug:
            return self.default_annotations_path
        return self.output_dir / "annotations" / f"{self.target_class}__{slug}.csv"

    def list_annotation_sets(self) -> list[str]:
        """Saved annotation-set names ('' is the default/unnamed set)."""

        names: list[str] = []
        if self.default_annotations_path.is_file():
            names.append("")
        folder = self.output_dir / "annotations"
        prefix = f"{self.target_class}__"
        if folder.is_dir():
            for path in sorted(folder.glob(f"{prefix}*.csv")):
                names.append(path.stem[len(prefix):])
        return names

    @property
    def session_path(self) -> Path:
        return self.output_dir / f"{self.target_class}.session.json"

    @property
    def polygons_path(self) -> Path:
        return self.output_dir / "polygons.json"


def candidate_public(cand: Candidate) -> dict[str, Any]:
    """Map-stage payload for one deduped candidate."""

    return {
        "row_key": cand.row_key,
        "lat": cand.lat,
        "lon": cand.lon,
        "score": cand.score,
        "recordist": cand.recordist,
        "has_coords": cand.lat is not None and cand.lon is not None,
    }


def item_public(item: SampledItem, *, debug: bool) -> dict[str, Any]:
    """Annotation-grid payload for one sampled item."""

    cand = item.candidate
    payload = {
        "row_key": cand.row_key,
        "png_filename": cand.png_filename,
        "image_exists": item.image_exists,
        "bucket_index": item.bucket_index,
    }
    if debug:
        payload.update(
            {
                "score": cand.score,
                "recordist": cand.recordist,
                "lat": cand.lat,
                "lon": cand.lon,
                "weight": item.weight,
                "rank_pct": item.rank_pct,
                "filename": cand.filename,
                "country": cand.metadata.get("country", ""),
                "date": cand.metadata.get("date", ""),
            }
        )
    return payload


def do_sample(state: AppState, polygon: list[list[float]] | None) -> dict[str, Any]:
    """Apply the geo filter + stratified sampling and update ``state``."""

    inside, outside = apply_geo_filter(state.candidates, polygon)
    items, summaries = stratified_sample(
        candidates=inside,
        n_buckets=state.n_buckets,
        samples_per_bucket=state.samples_per_bucket,
        seed=state.seed,
        spectrogram_dir=state.spectrogram_dir,
    )
    with state.lock:
        state.polygon = polygon
        state.items = items
        state.bucket_summaries = summaries
        state.filtered_scores = [c.score for c in inside]
        # keep any annotations whose row is still in the sample
        keep = state.items_by_key
        state.annotations = {k: v for k, v in state.annotations.items() if k in keep}
        # merge persisted annotations (the default set) for this sample
        for key, value in load_annotations(state.default_annotations_path).items():
            if key in keep:
                state.annotations.setdefault(key, value)

    return {
        "ok": True,
        "geo": {
            "included": len(inside),
            "excluded": len(outside),
            "total": len(state.candidates),
            "polygon": polygon,
        },
        "buckets": summaries,
        "items": [item_public(it, debug=state.debug_mode) for it in state.items],
        "annotations": dict(state.annotations),
        "missing_images": sum(1 for it in state.items if not it.image_exists),
        "debug_mode": state.debug_mode,
    }


def annotated_payload(state: AppState) -> list[dict[str, Any]]:
    """Build the list fed to ``compute_metrics`` from current annotations."""

    out = []
    for item in state.items:
        ann = state.annotations.get(item.candidate.row_key)
        if ann not in ANNOTATION_VALUES:
            continue
        out.append(
            {
                "bucket_index": item.bucket_index,
                "score": item.candidate.score,
                "weight": item.weight,
                "annotation": ann,
            }
        )
    return out


# ----------------------------------------------------------------------------
# Annotation persistence
# ----------------------------------------------------------------------------
def load_annotations(path: Path) -> dict[str, str]:
    """Load ``row_key -> annotation`` from a saved CSV."""

    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "row_key" not in reader.fieldnames:
            return {}
        for row in reader:
            key = (row.get("row_key") or "").strip()
            ann = (row.get("annotation") or "").strip().lower()
            if key and ann in ANNOTATION_VALUES:
                out[key] = ann
    return out


def save_annotations(state: AppState, name: str | None = None) -> tuple[int, Path]:
    """Write current annotations + a session sidecar.

    ``name`` selects a named annotation set (the unnamed set keeps the legacy
    filename).  Returns ``(rows_written, path)``.
    """

    path = state.annotations_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    items = state.items_by_key
    rows = []
    for key, ann in sorted(state.annotations.items()):
        item = items.get(key)
        if item is None or ann not in ANNOTATION_VALUES:
            continue
        cand = item.candidate
        rows.append(
            {
                "row_key": key,
                "annotation": ann,
                "annotated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "target_class": state.target_class,
                "filename": cand.filename,
                "png_filename": cand.png_filename,
                "xcid": cand.xcid,
                "clip_index": cand.clip_index,
                "recordist": cand.recordist,
                "lat": "" if cand.lat is None else f"{cand.lat:.6f}",
                "lon": "" if cand.lon is None else f"{cand.lon:.6f}",
                "score": f"{cand.score:.10g}",
                "bucket_index": item.bucket_index,
                "bucket_population": item.bucket_population,
                "bucket_sampled": item.bucket_sampled,
                "weight": f"{item.weight:.10g}",
                "rank_pct": f"{item.rank_pct:.6f}",
                "inference_csv": str(state.inference_csv),
            }
        )
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)

    session = {
        "target_class": state.target_class,
        "species_slug": state.species_slug,
        "inference_csv": str(state.inference_csv),
        "annotation_set": safe_name(name or ""),
        "seed": state.seed,
        "n_buckets": state.n_buckets,
        "samples_per_bucket": state.samples_per_bucket,
        "filters": [
            {"name": f.name, "threshold": f.threshold} for f in state.score_filters
        ],
        "polygon": state.polygon,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    state.session_path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    return len(rows), path


# ----------------------------------------------------------------------------
# HTTP server
# ----------------------------------------------------------------------------
class AnnotationServer(ThreadingHTTPServer):
    state: AppState


class AnnotationHandler(BaseHTTPRequestHandler):
    server: "AnnotationServer"

    def log_message(self, fmt: str, *args: Any) -> None:  # silence by default
        return

    # -- GET ----------------------------------------------------------------
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_html(build_html())
            return
        if path == "/api/init":
            self.send_json(self.init_payload())
            return
        if path.startswith("/spectrogram/"):
            self.send_spectrogram(unquote(path.removeprefix("/spectrogram/")))
            return
        if path.startswith("/figure/"):
            self.send_figure(unquote(path.removeprefix("/figure/")))
            return
        if path == "/api/polygons":
            self.send_json({"polygons": load_polygons(self.server.state.polygons_path)})
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    # -- POST ---------------------------------------------------------------
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/sample":
                body = self.read_json()
                self.send_json(do_sample(self.server.state, body.get("polygon")))
                return
            if path == "/api/annotate":
                self.send_json(self.handle_annotate(self.read_json()))
                return
            if path == "/api/save":
                self.send_json(self.handle_save(self.read_json()))
                return
            if path == "/api/load":
                self.send_json(self.handle_load(self.read_json()))
                return
            if path == "/api/save-polygon":
                self.send_json(self.handle_save_polygon(self.read_json()))
                return
            if path == "/api/metrics":
                self.send_json(self.handle_metrics())
                return
        except ValueError as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:  # noqa: BLE001 - surface to the browser
            self.send_json({"error": f"{type(exc).__name__}: {exc}"},
                           HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    # -- handlers -----------------------------------------------------------
    def init_payload(self) -> dict[str, Any]:
        state = self.server.state
        s = state.join_stats
        return {
            "target_class": state.target_class,
            "species_slug": state.species_slug,
            "inference_csv": str(state.inference_csv),
            "spectrogram_dir": str(state.spectrogram_dir),
            "output_dir": str(state.output_dir),
            "debug_mode": state.debug_mode,
            "n_buckets": state.n_buckets,
            "samples_per_bucket": state.samples_per_bucket,
            "seed": state.seed,
            "map": {"lat": MAP_CENTER_LAT, "lon": MAP_CENTER_LON, "zoom": MAP_ZOOM},
            "join_stats": {
                "total_rows": s.total_rows,
                "label_matched": s.label_matched,
                "joined": s.joined,
                "metadata_missing": s.metadata_missing,
                "after_filters": s.after_filters,
                "after_dedup": s.after_dedup,
            },
            "filters": [
                {"name": f.name, "threshold": f.threshold,
                 "excluded_below": f.excluded_below, "missing_score": f.missing_score}
                for f in state.score_filters
            ],
            "candidates": [candidate_public(c) for c in state.candidates],
            "annotation_sets": state.list_annotation_sets(),
            "polygons": load_polygons(state.polygons_path),
        }

    def handle_save(self, body: dict[str, Any]) -> dict[str, Any]:
        state = self.server.state
        name = str(body.get("name", "")).strip()
        with state.lock:
            saved, path = save_annotations(state, name)
            sets = state.list_annotation_sets()
        return {"ok": True, "saved": saved, "path": str(path),
                "name": safe_name(name), "annotation_sets": sets}

    def handle_load(self, body: dict[str, Any]) -> dict[str, Any]:
        state = self.server.state
        name = str(body.get("name", "")).strip()
        path = state.annotations_path(name)
        if not path.is_file():
            raise ValueError(f"No saved annotation set named '{name or '(default)'}'")
        loaded = load_annotations(path)
        keep = state.items_by_key
        applied = {k: v for k, v in loaded.items() if k in keep}
        with state.lock:
            state.annotations = dict(applied)
        return {"ok": True, "name": safe_name(name), "loaded_total": len(loaded),
                "applied": len(applied), "annotations": applied}

    def handle_save_polygon(self, body: dict[str, Any]) -> dict[str, Any]:
        state = self.server.state
        name = safe_name(str(body.get("name", "")))
        if not name:
            raise ValueError("Give the region a name before saving.")
        ring = body.get("polygon")
        if not isinstance(ring, list) or len(ring) < 3:
            raise ValueError("Draw a polygon (3+ points) before saving.")
        try:
            ring = [[float(p[0]), float(p[1])] for p in ring]
        except (TypeError, ValueError, IndexError):
            raise ValueError("Polygon must be a list of [lat, lon] points.")
        with state.lock:
            polygons = save_polygon(state.polygons_path, name, ring)
        return {"ok": True, "name": name, "polygons": polygons}

    def handle_annotate(self, body: dict[str, Any]) -> dict[str, Any]:
        state = self.server.state
        row_key = str(body.get("row_key", "")).strip()
        annotation = str(body.get("annotation", "")).strip().lower()
        if annotation not in ANNOTATION_VALUES and annotation != "clear":
            raise ValueError("annotation must be positive, negative, or clear")
        if row_key not in state.items_by_key:
            raise ValueError(f"unknown row_key: {row_key}")
        with state.lock:
            if annotation == "clear":
                state.annotations.pop(row_key, None)
            else:
                state.annotations[row_key] = annotation
        return {"ok": True, "row_key": row_key,
                "annotation": "" if annotation == "clear" else annotation}

    def handle_metrics(self) -> dict[str, Any]:
        state = self.server.state
        with state.lock:
            items = annotated_payload(state)
            summaries = list(state.bucket_summaries)
            scores = list(state.filtered_scores)
        if not summaries:
            raise ValueError("Sample first (draw a region or skip), then annotate.")
        if not items:
            raise ValueError("No annotations yet.")

        metrics = compute_metrics(
            annotated_items=items,
            bucket_summaries=summaries,
            n_buckets=state.n_buckets,
            prior_alpha=state.prior_alpha,
            prior_beta=state.prior_beta,
            bootstrap_samples=state.bootstrap_samples,
            seed=state.seed,
        )
        figures = render_figures(
            metrics=metrics,
            population_scores=scores,
            annotated_items=items,
            bucket_summaries=summaries,
            target_class=state.target_class,
            species_slug=state.species_slug,
            output_dir=state.output_dir,
        )
        # write metrics.json (without the heavy bootstrap arrays)
        clean = strip_samples(metrics)
        with state.lock:
            save_annotations(state)
        (state.output_dir / f"{state.target_class}.metrics.json").write_text(
            json.dumps(clean, indent=2), encoding="utf-8"
        )
        return {
            "ok": True,
            "metrics": clean,
            "figures": [Path(p).name for p in figures],
            "output_dir": str(state.output_dir),
        }

    # -- io helpers ---------------------------------------------------------
    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0:
            return {}
        if length > 5_000_000:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def send_spectrogram(self, filename: str) -> None:
        safe = Path(filename).name
        if not safe or safe != filename:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid filename")
            return
        image = self.server.state.spectrogram_dir / safe
        if not image.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        self._send_file(image, cache=True)

    def send_figure(self, filename: str) -> None:
        safe = Path(filename).name
        image = self.server.state.output_dir / safe
        if safe != filename or not image.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Figure not found")
            return
        self._send_file(image, cache=False)

    def _send_file(self, path: Path, *, cache: bool) -> None:
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(path.stat().st_size))
        if cache:
            self.send_header("Cache-Control", "public, max-age=3600")
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        with path.open("rb") as handle:
            shutil.copyfileobj(handle, self.wfile)

    def send_html(self, text: str) -> None:
        body = text.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def strip_samples(metrics: dict[str, Any]) -> dict[str, Any]:
    """Drop the big private bootstrap arrays before serialising."""

    clean = json.loads(json.dumps(metrics, default=lambda o: None))
    clean.get("density", {}).pop("_samples", None)
    roc = clean.get("roc_auc", {})
    roc.pop("_samples", None)
    roc.pop("_weighted_roc", None)
    roc.pop("_raw_roc", None)
    return clean


# ----------------------------------------------------------------------------
# Browser app
# ----------------------------------------------------------------------------
def build_html() -> str:
    """Return the single-page annotation UI."""

    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROC annotation</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.css"/>
<style>
  :root{--bg:#f5f6f8;--panel:#fff;--text:#16202b;--muted:#5d6b7b;--border:#d7dde5;
        --pos:#2b6f4e;--pos-bg:#e6f4ec;--neg:#7a2230;--neg-bg:#fbe9ec;--accent:#3b528b;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font:14px/1.4 Inter,system-ui,
       -apple-system,"Segoe UI",sans-serif}
  header{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.96);
         border-bottom:1px solid var(--border);backdrop-filter:blur(8px);padding:12px 18px}
  h1{margin:0;font-size:17px}
  .sub{color:var(--muted);font-size:12px;margin-top:2px;overflow-wrap:anywhere}
  .tabs{display:flex;gap:8px;margin-top:10px}
  .tab{padding:6px 12px;border:1px solid var(--border);border-radius:7px;background:#fff;
       cursor:pointer;font-weight:600}
  .tab.active{border-color:var(--accent);color:var(--accent);background:#eef1f8}
  main{padding:18px;max-width:1500px;margin:0 auto}
  .stage{display:none}
  .stage.active{display:block}
  .row{display:flex;gap:16px;flex-wrap:wrap;align-items:flex-start}
  #map{height:520px;flex:1 1 560px;min-width:340px;border:1px solid var(--border);
       border-radius:10px}
  .side{flex:1 1 300px;min-width:280px}
  .panel{background:var(--panel);border:1px solid var(--border);border-radius:10px;
         padding:14px;margin-bottom:14px}
  .panel h3{margin:0 0 8px;font-size:13px;text-transform:uppercase;letter-spacing:.04em;
            color:var(--muted)}
  .stat{display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px dashed #eceff3}
  .stat b{font-variant-numeric:tabular-nums}
  button{min-height:34px;border:1px solid var(--border);border-radius:8px;background:#fff;
         font:inherit;font-weight:600;cursor:pointer;padding:0 12px}
  button:hover{border-color:var(--accent)}
  button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
  input[type=text],select{min-height:34px;border:1px solid var(--border);border-radius:8px;
         background:#fff;font:inherit;padding:0 8px;color:var(--text)}
  input[type=text]{min-width:150px}
  .toolbar{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:14px}
  .toolbar .spacer{flex:1}
  table{width:100%;border-collapse:collapse;font-size:12px}
  th,td{text-align:right;padding:4px 6px;border-bottom:1px solid #edf0f3}
  th:first-child,td:first-child{text-align:left}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}
  .card{border:1px solid var(--border);border-radius:9px;background:#fff;overflow:hidden}
  .card.positive{border-color:var(--pos);box-shadow:0 0 0 2px rgba(43,111,78,.15)}
  .card.negative{border-color:var(--neg);box-shadow:0 0 0 2px rgba(122,34,48,.15)}
  .imgwrap{background:#eef1f4;display:flex;align-items:center;justify-content:center;
           min-height:120px;border-bottom:1px solid var(--border)}
  .imgwrap img{width:100%;display:block}
  .miss{padding:18px;color:var(--muted);font-size:12px;text-align:center}
  .cbody{padding:8px}
  .bstrip{height:6px;border-radius:3px;margin-bottom:6px}
  .dbg{font-size:11px;color:var(--muted);margin-top:6px;display:grid;
       grid-template-columns:1fr 1fr;gap:2px 8px;overflow-wrap:anywhere}
  .btns{display:grid;grid-template-columns:1fr 1fr 34px;gap:6px;margin-top:8px}
  .btns button{font-size:13px}
  button[data-v="positive"].sel{background:var(--pos-bg);border-color:var(--pos);color:var(--pos)}
  button[data-v="negative"].sel{background:var(--neg-bg);border-color:var(--neg);color:var(--neg)}
  .metricbig{display:flex;gap:18px;flex-wrap:wrap;margin-bottom:14px}
  .metricbig .m{background:#fff;border:1px solid var(--border);border-radius:10px;
                padding:12px 16px;min-width:170px}
  .metricbig .m b{display:block;font-size:24px;font-variant-numeric:tabular-nums}
  .metricbig .m span{color:var(--muted);font-size:12px}
  .figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}
  .figs img{width:100%;border:1px solid var(--border);border-radius:8px;background:#fff}
  .status{color:var(--muted);font-size:12px}
  .err{display:none;margin:8px 0;padding:8px 10px;border:1px solid #c0392b;border-radius:8px;
       background:#fdecea;color:#922}
  .legend{font-size:12px;color:var(--muted);margin-top:8px}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px;
       vertical-align:middle}
</style>
</head>
<body>
<header>
  <h1 id="title">ROC annotation</h1>
  <div class="sub" id="subtitle">loading…</div>
  <div class="tabs">
    <div class="tab active" data-stage="region">1 · Region</div>
    <div class="tab" data-stage="annotate">2 · Annotate</div>
    <div class="tab" data-stage="metrics">3 · Metrics</div>
  </div>
  <div class="err" id="err"></div>
</header>
<main>
  <!-- STAGE 1: REGION -->
  <section class="stage active" id="stage-region">
    <div class="toolbar">
      <button class="primary" id="sampleBtn">Sample &amp; annotate →</button>
      <button id="clearPoly">Clear polygon</button>
      <input id="regionName" type="text" placeholder="region name">
      <button id="saveRegion">Save region</button>
      <select id="regionSelect" title="saved regions"><option value="">— saved regions —</option></select>
      <button id="loadRegion">Load region</button>
      <span class="status" id="regionStatus"></span>
    </div>
    <div class="row">
      <div id="map"></div>
      <div class="side">
        <div class="panel">
          <h3>Region filter</h3>
          <div class="stat"><span>Deduped candidates</span><b id="cTotal">–</b></div>
          <div class="stat"><span>With coordinates</span><b id="cCoords">–</b></div>
          <div class="stat"><span>Inside polygon</span><b id="cIn">–</b></div>
          <div class="stat"><span>Excluded</span><b id="cOut">–</b></div>
          <div class="legend">
            <span class="dot" style="background:#21908d"></span>included
            <span class="dot" style="background:#bbbbbb;margin-left:10px"></span>excluded
          </div>
          <div class="legend">Draw a polygon with the toolbar (top-left of map).
            No polygon = keep everything with coordinates.</div>
        </div>
        <div class="panel">
          <h3>Join</h3>
          <div class="stat"><span>Inference rows (class)</span><b id="jLabel">–</b></div>
          <div class="stat"><span>Joined to metadata</span><b id="jJoined">–</b></div>
          <div class="stat"><span>Metadata missing</span><b id="jMiss">–</b></div>
          <div class="stat"><span>After score filters</span><b id="jFilt">–</b></div>
          <div class="stat"><span>After recordist dedup</span><b id="jDedup">–</b></div>
        </div>
      </div>
    </div>
  </section>

  <!-- STAGE 2: ANNOTATE -->
  <section class="stage" id="stage-annotate">
    <div class="toolbar">
      <input id="setName" type="text" placeholder="annotation set name">
      <button id="saveBtn">Save</button>
      <select id="setSelect" title="saved annotation sets"><option value="">— saved sets —</option></select>
      <button id="loadBtn">Load</button>
      <label style="display:flex;gap:6px;align-items:center;font-weight:600">
        <input type="checkbox" id="dbgToggle"> debug info</label>
      <label style="display:flex;gap:6px;align-items:center">
        <select id="viewFilter">
          <option value="all">all sampled</option>
          <option value="unann">unannotated</option>
          <option value="ann">annotated</option>
        </select></label>
      <span class="spacer"></span>
      <span class="status" id="annStatus"></span>
      <button class="primary" id="toMetrics">Compute metrics →</button>
    </div>
    <div class="panel">
      <h3>Buckets (log score-quantiles)</h3>
      <table id="bucketTable"><thead><tr>
        <th>bucket</th><th>score range</th><th>P(b)</th><th>population</th>
        <th>sampled</th><th>+</th><th>−</th></tr></thead><tbody></tbody></table>
    </div>
    <div class="grid" id="grid"></div>
  </section>

  <!-- STAGE 3: METRICS -->
  <section class="stage" id="stage-metrics">
    <div class="toolbar">
      <button class="primary" id="computeBtn">Compute / refresh</button>
      <span class="status" id="metricStatus"></span>
    </div>
    <div class="metricbig" id="metricBig"></div>
    <div class="figs" id="figs"></div>
  </section>
</main>

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet-draw@1.0.4/dist/leaflet.draw.js"></script>
<script>
const S={init:null,candidates:[],items:[],buckets:[],annotations:{},debug:false,
         polygon:null,polygons:{}};
const $=id=>document.getElementById(id);
function err(m){const e=$("err");e.style.display=m?"block":"none";e.textContent=m||"";}
function pct(x){return Number.isFinite(x)?(x*100).toFixed(1)+"%":"–";}
function f3(x){return Number.isFinite(x)?x.toFixed(3):"–";}
function f4(x){return Number.isFinite(x)?x.toFixed(4):"–";}

// ---- Leaflet map + draw ----
let map,pointLayer,drawn,fg;
const VIR={lo:"#440154",mid:"#21908d",hi:"#fde725"};
function pointInPoly(lat,lon,poly){
  let inside=false;for(let i=0,j=poly.length-1;i<poly.length;j=i++){
    const yi=poly[i][0],xi=poly[i][1],yj=poly[j][0],xj=poly[j][1];
    if(((yi>lat)!==(yj>lat))&&(lon<(xj-xi)*(lat-yi)/((yj-yi)||1e-12)+xi))inside=!inside;}
  return inside;}
function currentPolygon(){
  if(!drawn)return null;const ls=drawn.getLatLngs();
  const ring=Array.isArray(ls[0])?ls[0]:ls;return ring.map(p=>[p.lat,p.lng]);}
function recolor(){
  const poly=currentPolygon();let inN=0,outN=0,coords=0;
  pointLayer.eachLayer(l=>{
    const c=l.options._cand;if(c.has_coords)coords++;
    const inside=c.has_coords&&(!poly||pointInPoly(c.lat,c.lon,poly));
    if(inside){inN++;l.setStyle({color:"#21908d",fillColor:"#21908d",fillOpacity:.85});}
    else{outN++;l.setStyle({color:"#bbbbbb",fillColor:"#bbbbbb",fillOpacity:.5});}});
  $("cIn").textContent=inN;$("cOut").textContent=outN;$("cCoords").textContent=coords;
  S.polygon=poly;}
function setPolygon(ring){
  if(typeof L==="undefined"||!fg){err("Map unavailable (offline); cannot load region.");return;}
  fg.clearLayers();drawn=L.polygon(ring.map(p=>[p[0],p[1]]));fg.addLayer(drawn);
  recolor();try{map.fitBounds(drawn.getBounds(),{padding:[30,30]});}catch(e){}}
function fillSelect(sel,names,emptyLabel,labelFn){
  sel.innerHTML=`<option value="">${emptyLabel}</option>`+
    names.map(n=>`<option value="${encodeURIComponent(n)}">${labelFn?labelFn(n):n}</option>`).join("");}
function initMap(){
  if(typeof L==="undefined"){
    $("map").innerHTML='<div class="miss">Map library (Leaflet CDN) did not load — '+
      'you appear to be offline. The polygon filter is unavailable, but you can still '+
      'click “Sample &amp; annotate →” to use every candidate with coordinates.</div>';
    $("regionStatus").textContent="offline: no map; polygon filter disabled";
    return;}
  const m=S.init.map;map=L.map("map").setView([m.lat,m.lon],m.zoom);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {maxZoom:19,attribution:"© OpenStreetMap"}).addTo(map);
  pointLayer=L.layerGroup().addTo(map);
  fg=new L.FeatureGroup();map.addLayer(fg);
  const dc=new L.Control.Draw({draw:{polygon:{showArea:false},marker:false,
    polyline:false,rectangle:true,circle:false,circlemarker:false},edit:{featureGroup:fg}});
  map.addControl(dc);
  map.on(L.Draw.Event.CREATED,e=>{fg.clearLayers();drawn=e.layer;fg.addLayer(drawn);recolor();});
  map.on(L.Draw.Event.EDITED,recolor);
  map.on(L.Draw.Event.DELETED,()=>{drawn=null;recolor();});
  $("clearPoly").onclick=()=>{fg.clearLayers();drawn=null;recolor();};
  const pts=[];
  S.candidates.forEach(c=>{
    if(!c.has_coords)return;
    const cm=L.circleMarker([c.lat,c.lon],{radius:5,weight:1,_cand:c,
      color:"#21908d",fillColor:"#21908d",fillOpacity:.85});
    cm.bindTooltip(`${c.recordist||"?"} · logit ${f3(c.score)}`);
    cm.addTo(pointLayer);pts.push([c.lat,c.lon]);});
  if(pts.length){try{map.fitBounds(pts,{padding:[30,30]});}catch(e){}}
  recolor();
}

// ---- stages ----
function showStage(name){
  document.querySelectorAll(".tab").forEach(t=>t.classList.toggle("active",t.dataset.stage===name));
  document.querySelectorAll(".stage").forEach(s=>s.classList.remove("active"));
  $("stage-"+name).classList.add("active");
  if(name==="region"&&map)setTimeout(()=>map.invalidateSize(),50);}
document.querySelectorAll(".tab").forEach(t=>t.onclick=()=>showStage(t.dataset.stage));

async function getJSON(url,opt){const r=await fetch(url,opt);const j=await r.json();
  if(!r.ok)throw new Error(j.error||(url+" → "+r.status));return j;}

async function boot(){
  S.init=await getJSON("/api/init");
  S.candidates=S.init.candidates;S.debug=S.init.debug_mode;
  $("title").textContent=`ROC annotation — ${S.init.target_class}`;
  const fl=S.init.filters||[];
  const ftxt=fl.map(f=>` · filter ${f.name}≥${f.threshold}`).join("");
  $("subtitle").textContent=
    `${S.init.species_slug} · ${S.init.n_buckets} buckets × ${S.init.samples_per_bucket}/bucket · seed ${S.init.seed}${ftxt} · `+
    `${S.init.inference_csv}`;
  const j=S.init.join_stats;
  $("cTotal").textContent=S.candidates.length;
  $("jLabel").textContent=j.label_matched;$("jJoined").textContent=j.joined;
  $("jMiss").textContent=j.metadata_missing;
  $("jFilt").textContent=fl.length?j.after_filters:"–";
  $("jDedup").textContent=S.candidates.length;
  $("dbgToggle").checked=S.debug;
  S.polygons=S.init.polygons||{};
  refreshRegionSelect();
  refreshSetSelect(S.init.annotation_sets||[]);
  initMap();
}

function refreshRegionSelect(){
  fillSelect($("regionSelect"),Object.keys(S.polygons).sort(),"— saved regions —");}
function setLabel(n){return n===""?"(default)":n;}
function refreshSetSelect(names){
  fillSelect($("setSelect"),names,"— saved sets —",setLabel);}

// ---- saved regions ----
$("saveRegion").onclick=async()=>{
  err("");
  const name=$("regionName").value.trim();
  if(!name){err("Type a region name first.");return;}
  if(!S.polygon){err("Draw a polygon first.");return;}
  try{
    const res=await getJSON("/api/save-polygon",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({name,polygon:S.polygon})});
    S.polygons=res.polygons||{};refreshRegionSelect();
    $("regionSelect").value=encodeURIComponent(res.name);
    $("regionStatus").textContent=`saved region “${res.name}”`;
  }catch(e){err(e.message);}
};
$("loadRegion").onclick=()=>{
  err("");
  const name=decodeURIComponent($("regionSelect").value||"");
  if(!name||!S.polygons[name]){err("Pick a saved region to load.");return;}
  setPolygon(S.polygons[name]);
  $("regionName").value=name;
  $("regionStatus").textContent=`loaded region “${name}”`;
};

$("sampleBtn").onclick=async()=>{
  err("");$("regionStatus").textContent="sampling…";
  try{
    const res=await getJSON("/api/sample",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({polygon:S.polygon})});
    S.items=res.items;S.buckets=res.buckets;S.annotations=res.annotations||{};
    S.debug=res.debug_mode;$("dbgToggle").checked=S.debug;
    $("regionStatus").textContent=
      `included ${res.geo.included}, excluded ${res.geo.excluded}, `+
      `sampled ${res.items.length} (${res.missing_images} missing images)`;
    renderBuckets();renderGrid();showStage("annotate");
  }catch(e){err(e.message);$("regionStatus").textContent="";}
};

function renderBuckets(){
  const tb=$("bucketTable").querySelector("tbody");
  tb.innerHTML=S.buckets.map(b=>{
    const counts=bucketCounts(b.bucket_index);
    const range=(b.score_min==null)?"–":`${f3(b.score_min)} … ${f3(b.score_max)}`;
    return `<tr><td>b${b.bucket_index}</td><td>${range}</td><td>${f3(b.p_b)}</td>
      <td>${b.population}</td><td>${b.sampled}</td><td>${counts.pos}</td>
      <td>${counts.neg}</td></tr>`;}).join("");
}
function bucketCounts(bi){let pos=0,neg=0;S.items.forEach(it=>{
  if(it.bucket_index!==bi)return;const a=S.annotations[it.row_key];
  if(a==="positive")pos++;else if(a==="negative")neg++;});return{pos,neg};}

function bucketColor(bi){const n=Math.max(S.init.n_buckets-1,1);const t=bi/n;
  // simple viridis-ish ramp
  const stops=[[68,1,84],[59,82,139],[33,144,141],[93,201,99],[253,231,37]];
  const x=t*(stops.length-1),i=Math.floor(x),f=x-i;
  const a=stops[i],b=stops[Math.min(i+1,stops.length-1)];
  const c=a.map((v,k)=>Math.round(v+(b[k]-v)*f));return `rgb(${c[0]},${c[1]},${c[2]})`;}

function renderGrid(){
  const view=$("viewFilter").value;
  const cards=S.items.filter(it=>{const a=S.annotations[it.row_key];
    if(view==="ann")return!!a;if(view==="unann")return!a;return true;}).map(cardHtml).join("");
  $("grid").innerHTML=cards||`<div class="miss">No cards in this view.</div>`;
}
function cardHtml(it){
  const a=S.annotations[it.row_key]||"";
  const img=it.image_exists
    ?`<img loading="lazy" src="/spectrogram/${encodeURIComponent(it.png_filename)}" alt="">`
    :`<div class="miss">missing image<br>${it.png_filename}</div>`;
  let dbg="";
  if(S.debug){dbg=`<div class="dbg">
     <span>logit</span><span>${f3(it.score)}</span>
     <span>bucket</span><span>b${it.bucket_index}</span>
     <span>recordist</span><span>${it.recordist||"?"}</span>
     <span>weight</span><span>${f3(it.weight)}</span>
     <span>country</span><span>${it.country||""}</span>
     <span>date</span><span>${it.date||""}</span></div>`;}
  // The bucket colour strip encodes the score band, so it is shown only in
  // debug mode -- by default it stays hidden to keep score perception from
  // biasing annotation.
  const strip=S.debug?`<div class="bstrip" style="background:${bucketColor(it.bucket_index)}"></div>`:"";
  return `<article class="card ${a}" data-k="${it.row_key}">
    <div class="imgwrap">${img}</div>
    <div class="cbody">
      ${strip}
      <div class="btns">
        <button data-k="${it.row_key}" data-v="positive" class="${a==="positive"?"sel":""}">＋ pos</button>
        <button data-k="${it.row_key}" data-v="negative" class="${a==="negative"?"sel":""}">－ neg</button>
        <button data-k="${it.row_key}" data-v="clear">✕</button>
      </div>${dbg}
    </div></article>`;
}
$("viewFilter").onchange=renderGrid;
$("dbgToggle").onchange=()=>{S.debug=$("dbgToggle").checked;renderGrid();};

$("grid").addEventListener("click",async e=>{
  const b=e.target.closest("button[data-k]");if(!b)return;
  const k=b.dataset.k,v=b.dataset.v;
  try{
    const res=await getJSON("/api/annotate",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({row_key:k,annotation:v})});
    if(res.annotation)S.annotations[k]=res.annotation;else delete S.annotations[k];
    updateCard(k);renderBuckets();
    if($("viewFilter").value!=="all")renderGrid();
  }catch(ex){err(ex.message);}
});
function updateCard(k){const card=$("grid").querySelector(`[data-k="${CSS.escape(k)}"]`);
  if(!card)return;const a=S.annotations[k]||"";card.classList.remove("positive","negative");
  if(a)card.classList.add(a);
  card.querySelectorAll("button[data-v]").forEach(btn=>
    btn.classList.toggle("sel",btn.dataset.v===a));}

$("saveBtn").onclick=async()=>{
  err("");$("annStatus").textContent="saving…";
  const name=$("setName").value.trim();
  try{const res=await getJSON("/api/save",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({name})});
    refreshSetSelect(res.annotation_sets||[]);
    $("setSelect").value=encodeURIComponent(res.name);
    $("annStatus").textContent=`saved ${res.saved} → ${res.path}`;}
  catch(e){err(e.message);$("annStatus").textContent="";}
};
$("loadBtn").onclick=async()=>{
  err("");$("annStatus").textContent="loading…";
  const name=decodeURIComponent($("setSelect").value||"");
  try{const res=await getJSON("/api/load",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({name})});
    S.annotations=res.annotations||{};
    $("setName").value=res.name;
    renderBuckets();renderGrid();
    $("annStatus").textContent=`loaded ${res.applied} of ${res.loaded_total} (set “${setLabel(res.name)}”)`;}
  catch(e){err(e.message);$("annStatus").textContent="";}
};
$("toMetrics").onclick=()=>{showStage("metrics");compute();};
$("computeBtn").onclick=compute;

async function compute(){
  err("");$("metricStatus").textContent="computing (bootstrap + figures)…";
  try{
    const res=await getJSON("/api/metrics",{method:"POST"});
    const m=res.metrics;const d=m.density,r=m.roc_auc;
    $("metricBig").innerHTML=`
      <div class="m"><b>${f3(r.point)}</b><span>ROC-AUC (decomposition)<br>
        95% CI [${f3(r.ci95[0])}, ${f3(r.ci95[1])}]</span></div>
      <div class="m"><b>${f4(d.point)}</b><span>call density P(+)<br>
        95% CI [${f4(d.ci95[0])}, ${f4(d.ci95[1])}]</span></div>
      <div class="m"><b>${m.n_annotated}</b><span>${m.n_positive} pos · ${m.n_negative} neg</span></div>
      <div class="m"><b>${f3(r.weighted_empirical)}</b>
        <span>weighted empirical AUC<br>raw: ${f3(r.raw_empirical)}</span></div>`;
    const stamp=Date.now();
    $("figs").innerHTML=res.figures.map(f=>
      `<img src="/figure/${encodeURIComponent(f)}?t=${stamp}" alt="${f}">`).join("");
    $("metricStatus").textContent=`saved to ${res.output_dir}`;
  }catch(e){err(e.message);$("metricStatus").textContent="";}
}

boot().catch(e=>err(e.message));
</script>
</body>
</html>
"""


# ----------------------------------------------------------------------------
# CLI + startup
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT,
                   help="Drive root holding agile_inferences/, embeddings/, spectrograms/.")
    p.add_argument("--target-class", default=TARGET_CLASS,
                   help="Entry name; also the agile_inferences/<species>/<class>/ run folder.")
    p.add_argument("--species-slug", default=SPECIES_SLUG)
    p.add_argument("--inference-name", default=INFERENCE_NAME,
                   help="agile_inferences run folder (defaults to --target-class).")
    p.add_argument("--inference-csv", type=Path, default=INFERENCE_CSV,
                   help="Explicit inference CSV; overrides the derived agile_inferences path.")
    p.add_argument("--metadata-csv", type=Path, default=METADATA_CSV,
                   help="Explicit metadata CSV; default <data_root>/embeddings/<species>/metadata.csv.")
    p.add_argument("--spectrogram-dir", type=Path, default=SPECTROGRAM_DIR,
                   help="Explicit spectrogram dir; default <data_root>/spectrograms/<species>.")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                   help="Explicit output dir; default <data_root>/eval_roc/<species>.")
    p.add_argument("--label", default=LABEL_FILTER,
                   help="Optional: keep only rows whose label column equals this. "
                        "Off by default since the run folder already scopes the class.")
    p.add_argument("--filter", dest="filters", action="append", default=None,
                   metavar="CLASS[:THRESHOLD]",
                   help="Prune the candidate pool to clips scoring at or above "
                        "THRESHOLD (default 0) on another agile_inferences run of "
                        "this species, e.g. 'song_vs_call:0' to keep songs and drop "
                        "calls. Repeatable; a clip must pass every filter. Applied "
                        "before recordist dedup and the geographic filter.")
    p.add_argument("--score-column", default=SCORE_COLUMN)
    p.add_argument("--filename-column", default=FILENAME_COLUMN)
    p.add_argument("--label-column", default=LABEL_COLUMN)
    p.add_argument("--n-buckets", type=int, default=NUM_QUANTILE_BUCKETS)
    p.add_argument("--samples-per-bucket", type=int, default=SAMPLES_PER_BUCKET)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--no-dedup", action="store_true", help="Disable recordist de-duplication.")
    p.add_argument("--debug-mode", action="store_true", default=DEBUG_MODE,
                   help="Show logit + recordist + lat/lon on each card.")
    p.add_argument("--host", default=HOST)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--open", action="store_true", default=AUTO_OPEN, help="Open the UI in a browser.")
    p.add_argument("--selftest", action="store_true",
                   help="Run the pipeline headlessly with fabricated labels and exit.")
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """Derive (inference, metadata, spectrogram, output) paths from the data root.

    Any path given explicitly (CLI or config constant) is used verbatim; the
    rest follow the on-drive layout keyed by species and the run folder.
    """

    data_root = args.data_root.expanduser()
    species = args.species_slug
    run = args.inference_name or args.target_class
    inference = args.inference_csv or (
        data_root / "agile_inferences" / species / run / "inference.csv"
    )
    metadata = args.metadata_csv or (
        data_root / "embeddings" / species / "metadata.csv"
    )
    spectrograms = args.spectrogram_dir or (data_root / "spectrograms" / species)
    output = args.output_dir or (data_root / "eval_roc" / species)
    return (
        inference.expanduser(),
        metadata.expanduser(),
        spectrograms.expanduser(),
        output.expanduser(),
    )


def build_state(args: argparse.Namespace) -> AppState:
    """Load + join + dedup, returning ready-to-serve state."""

    inference_csv, metadata_csv, spectrogram_dir, output_dir = resolve_paths(args)
    if not inference_csv.is_file():
        raise SystemExit(
            f"Inference CSV not found: {inference_csv}{available_runs_hint(inference_csv)}\n"
            "  (set --target-class / --inference-name, or pass --inference-csv)"
        )
    require_readable_file(metadata_csv, "Metadata CSV")

    filters: list[ScoreFilter] = []
    for spec in (args.filters if args.filters is not None else list(SCORE_FILTERS)):
        name, threshold = parse_filter_spec(spec)
        filter_csv = (
            args.data_root.expanduser() / "agile_inferences" / args.species_slug
            / name / "inference.csv"
        )
        if not filter_csv.is_file():
            raise SystemExit(
                f"Filter inference CSV not found: {filter_csv}"
                f"{available_runs_hint(filter_csv)}"
            )
        filters.append(
            ScoreFilter(
                name=name,
                threshold=threshold,
                inference_csv=filter_csv,
                scores=load_filter_scores(
                    filter_csv,
                    score_column=args.score_column,
                    filename_column=args.filename_column,
                ),
            )
        )

    metadata_index = load_metadata_index(metadata_csv)
    candidates, stats = load_candidates(
        inference_csv=inference_csv,
        metadata_index=metadata_index,
        target_class=args.target_class,
        label_filter=args.label,
        score_column=args.score_column,
        filename_column=args.filename_column,
        label_column=args.label_column,
    )
    candidates = apply_score_filters(candidates, filters)
    stats.after_filters = len(candidates)
    if not args.no_dedup and DEDUP_BY_RECORDIST:
        recordists = {c.recordist for c in candidates if c.recordist}
        stats.recordists = len(recordists)
        candidates = dedup_by_recordist(candidates)
    stats.after_dedup = len(candidates)

    if not candidates:
        detail = (
            f"  rows read: {stats.total_rows}, label-matched: {stats.label_matched}, "
            f"joined to metadata: {stats.joined}, metadata-missing: {stats.metadata_missing}"
        )
        detail += "".join(
            f"\n  filter {f.name} (logit >= {f.threshold:g}): "
            f"excluded {f.excluded_below} below threshold, {f.missing_score} unscored"
            for f in filters
        )
        label_hint = (
            f"\n  (--label '{args.label}' may not match the file's label column)"
            if args.label is not None else ""
        )
        raise SystemExit(
            f"No candidates from {inference_csv}.\n{detail}{label_hint}"
        )

    return AppState(
        candidates=candidates,
        join_stats=stats,
        spectrogram_dir=spectrogram_dir,
        output_dir=output_dir,
        target_class=args.target_class,
        species_slug=args.species_slug,
        inference_csv=inference_csv,
        n_buckets=args.n_buckets,
        samples_per_bucket=args.samples_per_bucket,
        seed=args.seed,
        prior_alpha=BETA_PRIOR_ALPHA,
        prior_beta=BETA_PRIOR_BETA,
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        debug_mode=args.debug_mode,
        score_filters=filters,
    )


def run_selftest(state: AppState) -> None:
    """Headless end-to-end check: sample, fabricate labels, compute metrics."""

    import random

    print(f"Self-test for class '{state.target_class}'")
    print(f"  joined candidates:   {state.join_stats.joined}")
    for flt in state.score_filters:
        print(f"  filter {flt.name} (logit >= {flt.threshold:g}): "
              f"-{flt.excluded_below} below, -{flt.missing_score} unscored")
    if state.score_filters:
        print(f"  after filters:       {state.join_stats.after_filters}")
    print(f"  recordists:          {state.join_stats.recordists}")
    print(f"  after dedup:         {len(state.candidates)}")

    payload = do_sample(state, None)
    print(f"  sampled items:       {len(state.items)} "
          f"(missing images: {payload['missing_images']})")
    print("  buckets:")
    for b in state.bucket_summaries:
        rng = ("–" if b["score_min"] is None
               else f"{b['score_min']:.3f}..{b['score_max']:.3f}")
        print(f"    b{b['bucket_index']}: pop={b['population']:>4} "
              f"sampled={b['sampled']:>3} P(b)={b['p_b']:.3f} range={rng}")

    # Fabricate labels: higher logit -> more likely positive (sigmoid).
    rng = random.Random(state.seed)
    for item in state.items:
        prob = 1.0 / (1.0 + math.exp(-(item.candidate.score - 1.0)))
        state.annotations[item.candidate.row_key] = (
            "positive" if rng.random() < prob else "negative"
        )

    items = annotated_payload(state)
    metrics = compute_metrics(
        annotated_items=items,
        bucket_summaries=state.bucket_summaries,
        n_buckets=state.n_buckets,
        prior_alpha=state.prior_alpha,
        prior_beta=state.prior_beta,
        bootstrap_samples=state.bootstrap_samples,
        seed=state.seed,
    )
    d, r = metrics["density"], metrics["roc_auc"]
    print(f"  P(+):                {d['point']:.4f}  CI95 "
          f"[{d['ci95'][0]:.4f}, {d['ci95'][1]:.4f}]")
    print(f"  ROC-AUC (decomp):    {r['point']:.4f}  CI95 "
          f"[{r['ci95'][0]:.4f}, {r['ci95'][1]:.4f}]")
    print(f"  ROC-AUC (weighted):  {r['weighted_empirical']:.4f}")
    print(f"  ROC-AUC (raw):       {r['raw_empirical']:.4f}")
    print("Self-test OK (labels were fabricated; numbers are illustrative).")


def main() -> None:
    args = parse_args()
    state = build_state(args)

    if args.selftest:
        run_selftest(state)
        return

    server = AnnotationServer((args.host, args.port), AnnotationHandler)
    server.state = state
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/"
    print("ROC annotation server")
    print(f"  URL:            {url}")
    print(f"  Target class:   {state.target_class}")
    for flt in state.score_filters:
        print(f"  Filter:         {flt.name} logit >= {flt.threshold:g} "
              f"(-{flt.excluded_below} below, -{flt.missing_score} unscored)")
    print(f"  Candidates:     {len(state.candidates)} (after recordist dedup)")
    print(f"  Spectrograms:   {state.spectrogram_dir}")
    print(f"  Output:         {state.output_dir}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
