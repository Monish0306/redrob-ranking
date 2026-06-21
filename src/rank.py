"""
rank.py  —  Phase 7 (MAIN ENTRYPOINT)
--------------------------------------
The single script that produces the submission CSV.

USAGE (the exact command in your README):
  python src/rank.py \\
    --candidates data/candidates.jsonl \\
    --jd         data/job_description.txt \\
    --out        submission.csv

COMPUTE BUDGET (what gets timed in Stage 3):
  This script must complete in ≤ 5 minutes on CPU-only, 16GB RAM, no network.
  The embedding pre-compute step is in scripts/precompute_embeddings.py (separate).
  The only embedding this script does is the JD itself (one short text).

WHAT HAPPENS IN ORDER:
  1. Load candidate embeddings from disk (.npy artifact)
  2. Embed the JD text with local model
  3. Compute cosine similarity (vectorized, milliseconds)
  4. Load + validate all candidates
  5. Run honeypot filter
  6. Compute feature vectors
  7. Compute composite scores + availability multipliers
  8. Select top 100
  9. Generate reasoning strings
  10. Write CSV
  11. Print timing summary
"""

import sys
import time
import logging
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    CANDIDATES_FILE, EMBEDDINGS_FILE, CANDIDATE_IDS_FILE,
    JD_FILE, TOP_K, SUBMISSION_COLUMNS, EMBEDDING_MODEL,
)
from src.ingest import load_all_candidates
from src.honeypot_filter import filter_candidates, honeypot_summary
from src.features import compute_features_batch
from src.scoring import (
    EmbeddingIndex, embed_jd,
    score_all_candidates, select_top_k,
)
from src.reasoning import generate_reasoning_batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_jd(jd_path: Path) -> str:
    """Load job description text."""
    if not jd_path.exists():
        raise FileNotFoundError(f"JD file not found: {jd_path}")
    return jd_path.read_text(encoding="utf-8").strip()


def write_csv(
    top_candidates: list,
    reasonings: list,
    output_path: Path,
) -> None:
    """
    Write the submission CSV.
    Columns: candidate_id, rank, score, reasoning
    Exactly 100 rows. Scores non-increasing with rank (guaranteed by select_top_k).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for sc, reasoning in zip(top_candidates, reasonings):
            writer.writerow({
                "candidate_id": sc.candidate_id,
                "rank":         sc.rank,
                "score":        sc.final_score,
                "reasoning":    reasoning,
            })
    logger.info(f"✅ CSV written → {output_path} ({len(top_candidates)} rows)")


def run(
    candidates_path: Path,
    jd_path: Path,
    output_path: Path,
    embeddings_path: Path = EMBEDDINGS_FILE,
    ids_path: Path        = CANDIDATE_IDS_FILE,
    model_name: str       = EMBEDDING_MODEL,
    top_k: int            = TOP_K,
    max_records: int      = None,  # for dev/testing; None = all 100K
):
    t_total = time.time()
    timings = {}

    # ── Step 1: Load pre-computed embeddings ─────────────────────────────────
    t = time.time()
    index = EmbeddingIndex(embeddings_path, ids_path)
    index.load()
    timings["load_embeddings"] = time.time() - t

    # ── Step 2: Embed the JD (one text, fast) ────────────────────────────────
    t = time.time()
    jd_text      = load_jd(jd_path)
    jd_embedding = embed_jd(jd_text, model_name)
    timings["embed_jd"] = time.time() - t

    # ── Step 3: Cosine similarity (vectorized, milliseconds) ─────────────────
    t = time.time()
    semantic_scores = index.score_against_jd(jd_embedding)
    timings["cosine_similarity"] = time.time() - t
    logger.info(f"Semantic scores computed for {len(semantic_scores):,} candidates")

    # ── Step 4: Load + validate candidates ───────────────────────────────────
    t = time.time()
    candidates = load_all_candidates(
        candidates_path, validate=True, max_records=max_records
    )
    timings["load_candidates"] = time.time() - t

    # ── Step 5: Honeypot filter ───────────────────────────────────────────────
    t = time.time()
    clean, flagged = filter_candidates(candidates, verbose=False)
    hp_summary = honeypot_summary(flagged)
    timings["honeypot_filter"] = time.time() - t
    logger.info(
        f"Honeypot filter: {len(clean):,} clean, {len(flagged)} excluded "
        f"| rules: {hp_summary['rules_triggered']}"
    )

    # ── Step 6: Feature extraction ────────────────────────────────────────────
    t = time.time()
    feature_vectors = compute_features_batch(clean)
    timings["feature_extraction"] = time.time() - t

    # ── Step 7: Build signals lookup ──────────────────────────────────────────
    signals_map = {c["candidate_id"]: c["redrob_signals"] for c in clean}

    # ── Step 8: Composite scoring ─────────────────────────────────────────────
    t = time.time()
    scored = score_all_candidates(feature_vectors, semantic_scores, signals_map)
    timings["scoring"] = time.time() - t

    # ── Step 9: Select top-K ──────────────────────────────────────────────────
    t = time.time()
    top_k_candidates = select_top_k(scored, k=top_k)
    timings["top_k_selection"] = time.time() - t

    # ── Step 10: Generate reasoning ───────────────────────────────────────────
    t = time.time()
    reasonings = generate_reasoning_batch(top_k_candidates)
    timings["reasoning"] = time.time() - t

    # ── Step 11: Write CSV ────────────────────────────────────────────────────
    t = time.time()
    write_csv(top_k_candidates, reasonings, output_path)
    timings["write_csv"] = time.time() - t

    # ── Timing summary ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    logger.info("\n" + "="*55)
    logger.info("TIMING BREAKDOWN")
    logger.info("="*55)
    for step, elapsed in timings.items():
        logger.info(f"  {step:<25s}: {elapsed:5.1f}s")
    logger.info(f"  {'TOTAL':<25s}: {total_elapsed:5.1f}s")
    logger.info("="*55)

    if total_elapsed > 300:
        logger.warning(
            f"⚠️  Total time {total_elapsed:.0f}s exceeds 5-minute budget! "
            f"Optimize embedding or batch size."
        )
    else:
        logger.info(
            f"✅ Completed in {total_elapsed:.1f}s "
            f"({300-total_elapsed:.0f}s under 5-min budget)"
        )

    # ── Top-10 preview ────────────────────────────────────────────────────────
    logger.info("\nTOP 10 PREVIEW:")
    logger.info(f"{'Rank':<5} {'ID':<15} {'Score':<7} {'Title-Bucket':<10} "
                f"{'Skills[:2]'}")
    for sc in top_k_candidates[:10]:
        fv = sc.features
        logger.info(
            f"  #{sc.rank:<4} {sc.candidate_id:<15} {sc.final_score:<7.4f} "
            f"{fv.title_bucket:<10} {fv.matched_skills[:2]}"
        )

    return top_k_candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranking — produce submission.csv"
    )
    parser.add_argument("--candidates",  type=Path, default=CANDIDATES_FILE)
    parser.add_argument("--jd",          type=Path, default=JD_FILE)
    parser.add_argument("--out",         type=Path, default=Path("submission.csv"))
    parser.add_argument("--embeddings",  type=Path, default=EMBEDDINGS_FILE)
    parser.add_argument("--ids",         type=Path, default=CANDIDATE_IDS_FILE)
    parser.add_argument("--model",       type=str,  default=EMBEDDING_MODEL)
    parser.add_argument("--top-k",       type=int,  default=TOP_K)
    parser.add_argument("--max-records", type=int,  default=None)
    args = parser.parse_args()

    run(
        candidates_path = args.candidates,
        jd_path         = args.jd,
        output_path     = args.out,
        embeddings_path = args.embeddings,
        ids_path        = args.ids,
        model_name      = args.model,
        top_k           = args.top_k,
        max_records     = args.max_records,
    )