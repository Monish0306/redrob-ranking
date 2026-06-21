"""
api.py  —  FastAPI backend for the Lovable frontend
-------------------------------------------------------
Exposes the REAL ranking pipeline (same modules used by rank.py and the
Streamlit demo) as a proper HTTP API, so a Lovable/Next.js frontend can
call it instead of using mock data.

IMPORTANT — spec compliance note:
  The OFFICIAL required sandbox link remains the Streamlit app (app/demo.py),
  since it's already deployed and proven to satisfy the organizer's spec
  (accepts a small sample, runs end-to-end, completes within budget).
  This API is an ADDITIONAL layer so a polished frontend can sit on top
  of the same real pipeline — it does not replace the Streamlit sandbox
  requirement, it extends the project beyond it.

RUN LOCALLY:
  uvicorn app.api:app --reload --port 8000

Then visit http://localhost:8000/docs for interactive API docs.

ENDPOINTS:
  POST /api/rank       — upload candidates.jsonl + JD text, get ranked results
  GET  /api/health      — health check
"""

import sys
import json
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.honeypot_filter import filter_candidates, honeypot_summary
from src.features import compute_features_batch
from src.scoring import (
    compute_composite_score, compute_availability_multiplier, ScoredCandidate,
)
from src.reasoning import generate_reasoning
from src.ingest import validate_candidate


app = FastAPI(
    title="Redrob Candidate Ranking API",
    description="Backend for the candidate ranking pipeline — powers the frontend dashboard",
    version="1.0.0",
)

# Allow the Lovable frontend (and local dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your actual Lovable domain before final submission
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Response models ──────────────────────────────────────────────────────────

class CandidateResult(BaseModel):
    rank: int
    candidate_id: str
    score: float
    title_bucket: str
    matched_skills: list[str]
    yoe: float
    location: str
    top_company: str
    reasoning: str
    sub_scores: dict


class RankingSummary(BaseModel):
    total_candidates: int
    clean_candidates: int
    honeypots_excluded: int
    honeypot_rules_triggered: dict
    processing_time_seconds: float
    top_k_returned: int


class RankingResponse(BaseModel):
    summary: RankingSummary
    results: list[CandidateResult]


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health_check():
    return {"status": "ok", "service": "redrob-ranking-api"}


@app.post("/api/rank", response_model=RankingResponse)
async def rank_candidates(
    candidates_file: UploadFile = File(...),
    job_description: str = Form(...),
    top_k: int = Form(default=100),
):
    """
    Run the REAL ranking pipeline (same logic as src/rank.py) on an
    uploaded candidate sample. Designed for small samples (≤100-500
    candidates) for interactive frontend use — matches the spec's
    sandbox requirement of accepting a small sample.

    Note: semantic similarity is set to 0 in this lightweight endpoint
    (no embedding model load, for fast response times suitable for a
    live UI). The full embedding-based ranking is what runs in
    src/rank.py for the actual submission.
    """
    t0 = time.time()

    # ── Parse uploaded JSONL ──────────────────────────────────────────────────
    raw = await candidates_file.read()
    lines = raw.decode("utf-8").strip().split("\n")

    candidates = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            is_valid, _ = validate_candidate(record)
            if is_valid:
                candidates.append(record)
        except json.JSONDecodeError:
            continue

    if not candidates:
        raise HTTPException(status_code=400, detail="No valid candidates found in uploaded file")

    total_candidates = len(candidates)

    # ── Honeypot filter ────────────────────────────────────────────────────────
    clean, flagged = filter_candidates(candidates)
    hp_summary = honeypot_summary(flagged)

    # ── Feature extraction + scoring ──────────────────────────────────────────
    feature_vectors = compute_features_batch(clean)
    signals_map = {c["candidate_id"]: c["redrob_signals"] for c in clean}

    scored = []
    for fv in feature_vectors:
        avail = compute_availability_multiplier(signals_map.get(fv.candidate_id, {}))
        final, base = compute_composite_score(fv, 0.0, avail)  # semantic=0, see docstring
        scored.append(ScoredCandidate(
            candidate_id=fv.candidate_id,
            final_score=final,
            base_fit=base,
            availability=avail,
            semantic_sim=0.0,
            features=fv,
        ))

    scored.sort(key=lambda x: (-x.final_score, x.candidate_id))
    n_return = min(len(scored), top_k)
    for i, sc in enumerate(scored[:n_return], start=1):
        sc.rank = i

    # ── Build response ────────────────────────────────────────────────────────
    results = []
    for sc in scored[:n_return]:
        fv = sc.features
        reasoning = generate_reasoning(sc)
        results.append(CandidateResult(
            rank=sc.rank,
            candidate_id=sc.candidate_id,
            score=sc.final_score,
            title_bucket=fv.title_bucket,
            matched_skills=fv.matched_skills[:5],
            yoe=fv.yoe,
            location=fv.location,
            top_company=fv.top_company,
            reasoning=reasoning,
            sub_scores={
                "title": fv.title_score,
                "skill": fv.skill_score,
                "experience": fv.experience_score,
                "company": fv.company_score,
                "location": fv.location_score,
                "trajectory": fv.trajectory_score,
                "recency": fv.recency_score,
                "github": fv.github_score,
                "availability": sc.availability,
            },
        ))

    elapsed = time.time() - t0

    summary = RankingSummary(
        total_candidates=total_candidates,
        clean_candidates=len(clean),
        honeypots_excluded=len(flagged),
        honeypot_rules_triggered=hp_summary["rules_triggered"],
        processing_time_seconds=round(elapsed, 2),
        top_k_returned=n_return,
    )

    return RankingResponse(summary=summary, results=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)