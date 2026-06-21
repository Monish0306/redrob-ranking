"""
scoring.py  —  Phase 5
-----------------------
Composite scoring engine.
Combines all feature dimensions + semantic similarity + behavioral multiplier.

FORMULA:
  base_fit     = weighted sum of feature dimensions
  final_score  = base_fit × availability_multiplier

WHY multiplicative (not additive) for availability:
  The JD says inactive candidates should be "down-weighted appropriately."
  Additive: a great candidate with low availability still scores almost the same.
  Multiplicative: availability discounts the ENTIRE fit score.
  A 0.95-fit candidate who hasn't been active in 6 months becomes ~0.62 — 
  still in the top 100 likely, but not above an 0.80-fit active candidate.

YOUR ENHANCEMENT IDEAS:
  - "Adaptive weighting": if a candidate has VERY high semantic similarity
    but low title score, slightly up-weight semantic for that candidate only
    (catches the "hidden gem plain language" case the JD specifically named)
  - "Peer percentile bonus": if a candidate's skill_score is in the top 5%
    of all candidates, give a small bonus — they're genuinely exceptional
  - Add a "consistency bonus": if title, skill, and semantic all agree
    (all > 0.7), award a small bonus for coherent signal
"""

import logging
import numpy as np
from datetime import date, datetime
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.config import (
    WEIGHTS, AVAILABILITY_MIN, AVAILABILITY_MAX,
    ACTIVE_RECENT_DAYS, ACTIVE_STALE_DAYS,
    GITHUB_NO_ACCOUNT, SCORE_DECIMALS,
    EMBEDDINGS_FILE, CANDIDATE_IDS_FILE,
)
from src.features import FeatureVector

logger = logging.getLogger(__name__)
TODAY = date(2026, 6, 18)


# ─── Behavioral Availability Multiplier ──────────────────────────────────────
# PURPOSE: Discount base_fit for candidates who aren't actually reachable.
# 23 signals → compressed to one multiplier in [0.40, 1.0]
#
# Signal contributions (each normalized to their own range, then weighted):
#   open_to_work_flag (40%): most direct signal
#   last_active_date  (25%): recency of engagement
#   recruiter_response_rate (15%): historical responsiveness
#   interview_completion_rate (10%): follow-through
#   notice_period_days (10%): how fast they can join

def compute_availability_multiplier(signals: Dict[str, Any]) -> float:
    """
    Returns a multiplier in [AVAILABILITY_MIN, AVAILABILITY_MAX].
    Never returns 0 — even the worst available candidate shouldn't be zeroed.
    """
    score = 0.0

    # 1. Open to work flag (40% of multiplier)
    if signals.get("open_to_work_flag", False):
        score += 0.40
    else:
        score += 0.10  # not zero — could still respond to outreach

    # 2. Last active recency (25%)
    try:
        last_active = datetime.strptime(
            signals.get("last_active_date", "2020-01-01"), "%Y-%m-%d"
        ).date()
        days_since = (TODAY - last_active).days
        if days_since <= ACTIVE_RECENT_DAYS:
            score += 0.25
        elif days_since <= ACTIVE_STALE_DAYS:
            score += 0.25 * (1 - (days_since - ACTIVE_RECENT_DAYS) /
                             (ACTIVE_STALE_DAYS - ACTIVE_RECENT_DAYS))
        else:
            score += 0.03  # very stale
    except (ValueError, TypeError):
        score += 0.10

    # 3. Recruiter response rate (15%)
    rr = signals.get("recruiter_response_rate", 0.5)
    score += 0.15 * float(rr)

    # 4. Interview completion rate (10%)
    icr = signals.get("interview_completion_rate", 0.5)
    score += 0.10 * float(icr)

    # 5. Notice period (10%) — shorter is better for a hiring company
    notice = signals.get("notice_period_days", 90)
    if notice <= 15:
        score += 0.10
    elif notice <= 30:
        score += 0.08
    elif notice <= 60:
        score += 0.05
    elif notice <= 90:
        score += 0.03
    else:
        score += 0.01

    # Clamp to configured range
    return round(
        max(AVAILABILITY_MIN, min(AVAILABILITY_MAX, score)),
        4
    )


# ─── Semantic Similarity (embedding cosine) ───────────────────────────────────

class EmbeddingIndex:
    """
    Loads pre-computed candidate embeddings and provides fast cosine similarity.
    At ranking time: load once, query once (JD vs all candidates).
    No network calls. No model inference on candidates.
    """
    def __init__(
        self,
        embeddings_path: Path = EMBEDDINGS_FILE,
        ids_path: Path        = CANDIDATE_IDS_FILE,
    ):
        self.embeddings: Optional[np.ndarray] = None
        self.ids: Optional[np.ndarray] = None
        self._loaded = False
        self._paths  = (embeddings_path, ids_path)

    def load(self):
        emb_path, ids_path = self._paths
        if not emb_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {emb_path}\n"
                f"Run: python scripts/precompute_embeddings.py"
            )
        logger.info(f"Loading embeddings from {emb_path} ...")
        self.embeddings = np.load(emb_path)   # shape: (N, dim)
        self.ids        = np.load(ids_path)    # shape: (N,)
        logger.info(f"Loaded {len(self.ids):,} embeddings {self.embeddings.shape}")
        self._loaded = True

    def score_against_jd(self, jd_embedding: np.ndarray) -> Dict[str, float]:
        """
        Vectorized cosine similarity: JD vs all candidates.
        Returns dict: candidate_id → similarity_score [0, 1].
        Embeddings are L2-normalized at pre-compute time → dot product = cosine sim.
        """
        if not self._loaded:
            self.load()

        jd_norm = jd_embedding / (np.linalg.norm(jd_embedding) + 1e-10)
        similarities = self.embeddings @ jd_norm   # shape: (N,)
        # Shift from [-1, 1] to [0, 1]
        similarities = (similarities + 1.0) / 2.0

        return {
            str(cid): float(sim)
            for cid, sim in zip(self.ids, similarities)
        }


def embed_jd(jd_text: str, model_name: str) -> np.ndarray:
    """
    Embed the JD text. Called ONCE per ranking run.
    Model is loaded from local cache — HF_HUB_OFFLINE forces zero network
    calls, satisfying the no-network sandbox constraint exactly.
    """
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"        # block all HF Hub HTTP calls
    os.environ["TRANSFORMERS_OFFLINE"] = "1"   # block transformers lib network calls

    from sentence_transformers import SentenceTransformer
    logger.info(f"Embedding JD text ({len(jd_text)} chars) — offline mode")
    model = SentenceTransformer(model_name)
    embedding = model.encode(
        jd_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding


# ─── Composite Score ──────────────────────────────────────────────────────────

@dataclass
class ScoredCandidate:
    candidate_id:         str
    final_score:          float
    base_fit:             float
    availability:         float
    semantic_sim:         float
    features:             FeatureVector
    rank:                 int = 0   # filled in after sorting


def compute_composite_score(
    fv: FeatureVector,
    semantic_sim: float,
    availability: float,
) -> Tuple[float, float]:
    """
    Returns (final_score, base_fit).

    Enhancement: Adaptive weighting for hidden-gem detection.
      If semantic_sim is high but title is off-target, we slightly
      boost semantic weight — this is the "plain language" case the JD named.
    """
    w = WEIGHTS.copy()

    # ENHANCEMENT: Hidden-gem adaptive reweighting
    # If semantic similarity is very strong but title is off/adjacent,
    # reallocate some weight from title to semantic
    if semantic_sim >= 0.75 and fv.title_bucket in ("adjacent", "off"):
        transfer = 0.05
        w["title"]    = max(0.05, w["title"] - transfer)
        w["semantic"] = w["semantic"] + transfer

    # ENHANCEMENT: Consistency bonus
    # If title, skill, and semantic all agree (all > 0.7), small bonus
    consistency_bonus = 0.0
    if fv.title_score >= 0.7 and fv.skill_score >= 0.7 and semantic_sim >= 0.7:
        consistency_bonus = 0.03

    base_fit = (
        w["title"]      * fv.title_score      +
        w["skill"]      * fv.skill_score      +
        w["semantic"]   * semantic_sim         +
        w["experience"] * fv.experience_score  +
        w["company"]    * fv.company_score     +
        w["location"]   * fv.location_score   +
        # Enhancement dimensions (weighted lightly, additive)
        0.03 * fv.trajectory_score +
        0.02 * fv.recency_score    +
        0.01 * fv.github_score     +
        consistency_bonus
    )

    # Hard floor: if disqualified by JD criteria, cap at 0.10
    if fv.disqualified:
        base_fit = min(base_fit, 0.10)

    base_fit = max(0.0, min(1.0, base_fit))

    # Multiplicative availability discount
    final_score = round(base_fit * availability, SCORE_DECIMALS)

    return final_score, round(base_fit, SCORE_DECIMALS)


def score_all_candidates(
    feature_vectors: List[FeatureVector],
    semantic_scores: Dict[str, float],
    candidates_signals: Dict[str, Dict],
) -> List[ScoredCandidate]:
    """
    Score every candidate. Returns list of ScoredCandidate, unsorted.

    Args:
        feature_vectors:    From features.py
        semantic_scores:    {candidate_id: cosine_sim} from EmbeddingIndex
        candidates_signals: {candidate_id: redrob_signals dict}
    """
    results = []
    for fv in feature_vectors:
        cid = fv.candidate_id

        # Semantic similarity (0 if candidate wasn't embedded — e.g., was filtered)
        sem_sim = semantic_scores.get(cid, 0.0)

        # Availability from behavioral signals
        signals  = candidates_signals.get(cid, {})
        avail    = compute_availability_multiplier(signals)

        # Composite
        final, base = compute_composite_score(fv, sem_sim, avail)

        results.append(ScoredCandidate(
            candidate_id  = cid,
            final_score   = final,
            base_fit      = base,
            availability  = avail,
            semantic_sim  = sem_sim,
            features      = fv,
        ))

    return results


def select_top_k(
    scored: List[ScoredCandidate],
    k: int = 100,
) -> List[ScoredCandidate]:
    """
    Sort by final_score desc, tie-break by candidate_id asc (per spec).
    Assign ranks 1..k. Return top k.
    """
    sorted_candidates = sorted(
        scored,
        key=lambda x: (-x.final_score, x.candidate_id)
    )
    top_k = sorted_candidates[:k]
    for i, sc in enumerate(top_k, start=1):
        sc.rank = i
    return top_k


if __name__ == "__main__":
    # Smoke test without embeddings (semantic_sim = 0 for all)
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.ingest import load_all_candidates
    from src.honeypot_filter import filter_candidates
    from src.features import compute_features_batch

    data_path = Path("data/candidates.jsonl")
    print("Running scoring smoke test (no embeddings — semantic_sim=0)...")

    candidates = load_all_candidates(data_path, validate=True, max_records=2000)
    clean, _   = filter_candidates(candidates)
    fvecs      = compute_features_batch(clean)

    # Mock semantic scores (zeros) and signals lookup
    sem_scores = {c["candidate_id"]: 0.0 for c in clean}
    signals_map = {c["candidate_id"]: c["redrob_signals"] for c in clean}

    scored = score_all_candidates(fvecs, sem_scores, signals_map)
    top100 = select_top_k(scored, k=10)

    print(f"\nTop 10 (no semantic, features only):")
    for sc in top100:
        fv = sc.features
        print(f"  #{sc.rank} {sc.candidate_id} | {fv.title_bucket:8s} | "
              f"final={sc.final_score:.4f} base={sc.base_fit:.4f} avail={sc.availability:.2f} "
              f"| title={fv.title_score:.2f} skill={fv.skill_score:.2f} "
              f"| {fv.matched_skills[:2]}")