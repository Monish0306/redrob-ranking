"""
demo.py  —  Phase 10 (SANDBOX DEMO)
-------------------------------------
Small Streamlit app satisfying the organizer's required sandbox link.
Accepts a SMALL sample (≤100 candidates) and runs the full pipeline,
showing the ranked output live in a browser.

RUN LOCALLY:
  streamlit run app/demo.py

DEPLOY (free, public link):
  1. Push this repo to GitHub
  2. Go to share.streamlit.io
  3. Connect your GitHub repo
  4. Set main file path to: app/demo.py
  5. Deploy — you get a public URL like:
     https://your-app-name.streamlit.app

NOTE: This demo only needs to handle a small sample per spec —
it does NOT need to process the full 100K file.
"""

import sys
import json
import time
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.honeypot_filter import filter_candidates, honeypot_summary
from src.features import compute_features_batch
from src.scoring import (
    compute_composite_score, compute_availability_multiplier,
)
from src.reasoning import generate_reasoning
from src.scoring import ScoredCandidate
from src.ingest import validate_candidate


st.set_page_config(
    page_title="Redrob Candidate Ranker — Sandbox Demo",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Redrob Intelligent Candidate Discovery — Sandbox Demo")
st.caption(
    "India Runs Hackathon · Track 01 · Upload a small candidate sample "
    "(JSONL, ≤100 candidates) and a job description to see the ranking "
    "pipeline run live, end-to-end, with zero hosted-LLM calls."
)

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown("""
    This demo runs the **exact same pipeline** as the full submission:
    1. **Honeypot filter** — rule-based exclusion of impossible profiles
    2. **Feature extraction** — title/role gate, skill evidence, experience
       band, company type, location, trajectory, recency, GitHub signal
    3. **Composite scoring** — weighted combination × behavioral availability
       multiplier
    4. **Reasoning generation** — deterministic template, grounded in the
       same sub-scores that drove the rank — **zero LLM calls**

    For the full 100K-candidate run, semantic similarity uses pre-computed
    local embeddings (`sentence-transformers/all-MiniLM-L6-v2`). This demo
    skips that step for speed — semantic score defaults to 0 here, so the
    deterministic features (title, skill evidence, etc.) drive the ranking.
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload candidates (JSONL)")
    candidates_file = st.file_uploader(
        "candidates_sample.jsonl", type=["jsonl"],
        help="One JSON object per line, ≤100 candidates recommended",
    )

with col2:
    st.subheader("2. Paste Job Description")
    jd_text = st.text_area(
        "Job description text", height=200,
        placeholder="Paste the JD text here...",
    )

run_button = st.button("🚀 Run Ranking Pipeline", type="primary")

if run_button:
    if not candidates_file:
        st.error("Please upload a candidates JSONL file.")
        st.stop()
    if not jd_text.strip():
        st.error("Please paste a job description.")
        st.stop()

    t0 = time.time()

    # ── Parse uploaded JSONL ──────────────────────────────────────────────────
    with st.spinner("Parsing candidates..."):
        raw_lines = candidates_file.read().decode("utf-8").strip().split("\n")
        candidates = []
        parse_errors = 0
        for line in raw_lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                is_valid, errors = validate_candidate(record)
                if is_valid:
                    candidates.append(record)
                else:
                    parse_errors += 1
            except json.JSONDecodeError:
                parse_errors += 1

    if not candidates:
        st.error("No valid candidates found in the uploaded file.")
        st.stop()

    st.success(f"Loaded {len(candidates)} valid candidates "
               f"({parse_errors} skipped due to schema errors)")

    # ── Honeypot filter ────────────────────────────────────────────────────────
    with st.spinner("Running honeypot filter..."):
        clean, flagged = filter_candidates(candidates)
        summary = honeypot_summary(flagged)

    st.info(f"🍯 Honeypot filter: **{len(clean)}** clean, "
            f"**{len(flagged)}** excluded")
    if flagged:
        with st.expander(f"View {len(flagged)} excluded candidates"):
            for f in flagged:
                st.write(f"**{f['candidate_id']}** ({f['current_title']}): "
                        f"{', '.join(f['flags'])}")

    # ── Feature extraction + scoring (no embeddings in demo mode) ───────────────
    with st.spinner("Extracting features and scoring..."):
        feature_vectors = compute_features_batch(clean)
        signals_map = {c["candidate_id"]: c["redrob_signals"] for c in clean}

        scored = []
        for fv in feature_vectors:
            avail = compute_availability_multiplier(
                signals_map.get(fv.candidate_id, {})
            )
            # semantic_sim = 0 in demo mode (no embedding model loaded for speed)
            final, base = compute_composite_score(fv, 0.0, avail)
            scored.append(ScoredCandidate(
                candidate_id=fv.candidate_id,
                final_score=final,
                base_fit=base,
                availability=avail,
                semantic_sim=0.0,
                features=fv,
            ))

        scored.sort(key=lambda x: (-x.final_score, x.candidate_id))
        top_n = min(len(scored), 100)
        for i, sc in enumerate(scored[:top_n], start=1):
            sc.rank = i

    elapsed = time.time() - t0

    # ── Generate reasoning ────────────────────────────────────────────────────
    with st.spinner("Generating reasoning..."):
        results = []
        for sc in scored[:top_n]:
            reasoning = generate_reasoning(sc)
            fv = sc.features
            results.append({
                "Rank": sc.rank,
                "Candidate ID": sc.candidate_id,
                "Score": sc.final_score,
                "Title Bucket": fv.title_bucket,
                "Top Skills": ", ".join(fv.matched_skills[:3]),
                "Reasoning": reasoning,
            })

    df = pd.DataFrame(results)

    st.success(f"✅ Ranking complete in {elapsed:.2f}s for {len(clean)} candidates")

    st.subheader(f"📊 Top {top_n} Ranked Candidates")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Ranked CSV", data=csv_bytes,
        file_name="demo_ranking_output.csv", mime="text/csv",
    )

else:
    st.info("👆 Upload a candidates JSONL file and paste a JD, "
            "then click **Run Ranking Pipeline**.")

st.divider()
st.caption(
    "Redrob × India Runs Hackathon — Track 01 — Intelligent Candidate "
    "Discovery. Full submission pipeline: `src/rank.py`. "
    "No hosted-LLM API calls anywhere in this pipeline."
)