"""
precompute_embeddings.py  —  Phase 4 (OFFLINE STEP)
----------------------------------------------------
Run this ONCE, before the timed ranking step.
Saves candidate embeddings as .npy artifacts to disk.

WHY this is a separate script (not inside rank.py):
  - Embedding 100K candidates on CPU takes ~10-20 minutes
  - The ranking step must complete in ≤ 5 minutes
  - The organizer spec explicitly allows offline pre-computation
  - At ranking time, we only embed ONE text (the JD) and do a
    vectorized cosine similarity lookup — takes milliseconds

RUN: python scripts/precompute_embeddings.py

YOUR ENHANCEMENT IDEAS for this step:
  - Use a slightly larger model (e5-base-v2 instead of MiniLM) if
    you have 20-30 min to spare — better recall on semantic matches
  - Save two embedding matrices: one for headline+summary,
    one for career_history descriptions — dual-tower matching
  - Add a BM25 index alongside embeddings for true hybrid retrieval
    (install rank_bm25: pip install rank_bm25)
"""

import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import load_all_candidates
from src.honeypot_filter import filter_candidates
from src.config import (
    CANDIDATES_FILE, EMBEDDINGS_FILE, CANDIDATE_IDS_FILE,
    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, ARTIFACTS_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_candidate_text(candidate: dict) -> str:
    """
    Combine the most informative text fields into one string for embedding.

    WHY these fields:
      - headline: concise self-description, very dense signal
      - summary: their own narrative, catches "plain language" hidden gems
      - recent career descriptions (last 2 roles): actual work they did
      - top skills by name: reinforces domain vocabulary

    YOUR IDEA: Add education institution names (IIT/NIT bonus)
               or certification names to the embedded text.
    """
    p       = candidate["profile"]
    history = candidate.get("career_history", [])

    parts = []

    # Profile narrative
    if p.get("headline"):
        parts.append(p["headline"])
    if p.get("summary"):
        parts.append(p["summary"])

    # Last 2 career descriptions (most recent work is most relevant)
    for h in history[:2]:
        desc = h.get("description", "")
        if desc:
            parts.append(desc[:500])   # cap at 500 chars per role

    # Skill names (helps semantic match for domain vocabulary)
    skill_names = [s["name"] for s in candidate.get("skills", [])[:15]]
    if skill_names:
        parts.append("Skills: " + ", ".join(skill_names))

    return " ".join(parts)


def precompute(
    candidates_path: Path,
    embeddings_path: Path,
    ids_path: Path,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
    max_records: int = None,
):
    t_start = time.time()

    # Step 1: Load and filter
    logger.info(f"Loading candidates from {candidates_path} ...")
    candidates = load_all_candidates(candidates_path, validate=True,
                                     max_records=max_records)
    logger.info(f"Loaded {len(candidates):,} candidates")

    clean, flagged = filter_candidates(candidates)
    logger.info(f"After honeypot filter: {len(clean):,} clean candidates")

    # Step 2: Build text corpus
    logger.info("Building text corpus for embedding...")
    texts = []
    ids   = []
    for c in clean:
        texts.append(build_candidate_text(c))
        ids.append(c["candidate_id"])

    logger.info(f"Text corpus ready: {len(texts):,} documents")

    # Step 3: Load embedding model (downloads once, cached locally)
    logger.info(f"Loading model: {model_name}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    logger.info("Model loaded. Starting batch encoding...")

    # Step 4: Encode in batches with progress logging
    t_encode = time.time()
    embeddings = model.encode(
        texts,
        batch_size       = batch_size,
        show_progress_bar= True,
        normalize_embeddings = True,   # L2-normalized → cosine sim = dot product
        convert_to_numpy = True,
    )
    logger.info(f"Encoding done in {time.time()-t_encode:.1f}s")
    logger.info(f"Embedding matrix shape: {embeddings.shape}")

    # Step 5: Save artifacts
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    np.save(ids_path, np.array(ids))
    logger.info(f"✅ Embeddings saved → {embeddings_path}")
    logger.info(f"✅ Candidate IDs saved → {ids_path}")

    total_time = time.time() - t_start
    logger.info(f"Total precompute time: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info("This step is OUTSIDE the 5-minute ranking budget. ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-compute candidate embeddings")
    parser.add_argument("--candidates", type=Path, default=CANDIDATES_FILE)
    parser.add_argument("--out-embeddings", type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--out-ids", type=Path, default=CANDIDATE_IDS_FILE)
    parser.add_argument("--model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=EMBEDDING_BATCH_SIZE)
    parser.add_argument("--max-records", type=int, default=None,
                        help="Limit records for testing (default: all)")
    args = parser.parse_args()

    precompute(
        candidates_path  = args.candidates,
        embeddings_path  = args.out_embeddings,
        ids_path         = args.out_ids,
        model_name       = args.model,
        batch_size       = args.batch_size,
        max_records      = args.max_records,
    )