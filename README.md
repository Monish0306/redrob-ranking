# Redrob Intelligent Candidate Discovery — Ranking System

**India Runs Hackathon · Track 01 · The Data & AI Challenge**

A multi-signal candidate ranking pipeline that ranks 100,000 candidate
profiles against a single job description, returning the top 100 best-fit
candidates with deterministic, evidence-grounded reasoning — built to defeat
keyword-stuffing, catch honeypot profiles, and respect a strict CPU-only,
no-network, 5-minute compute budget.

---

## Quick Start (Reproduce the Submission)

### 1. Setup

```bash
git clone https://github.com/Monish0306/redrob-ranking.git
cd redrob-ranking

conda create -n redrob python=3.11 -y
conda activate redrob

pip install -r requirements.txt
```

### 2. Place the data

```
data/candidates.jsonl         # the 100K candidate dataset (not committed — too large)
data/job_description.txt      # the JD text (extracted from job_description.docx)
```

### 3. Pre-compute embeddings (one-time, offline step — outside the 5-min budget)

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
python scripts/precompute_embeddings.py
```

This saves `artifacts/candidate_embeddings.npy` and `artifacts/candidate_ids.npy`.
Per the submission spec, this step is explicitly allowed to exceed the
5-minute budget since it is a one-time, offline pre-computation.

### 4. Run the ranking pipeline (the timed step — must be ≤5 min)

```bash
python -m src.rank --candidates data/candidates.jsonl --jd data/job_description.txt --out submission.csv
```

### 5. Validate

```bash
python validate_submission.py submission.csv
```

Expected output: `Submission is valid.`

---

## Architecture

```
candidates.jsonl (100,000 profiles)
            │
            ▼
    1. Ingestion & schema validation     (src/ingest.py)
            │
            ▼
    2. Honeypot pre-filter                (src/honeypot_filter.py)
       — rule-based, excludes ~65 candidates with impossible profiles
            │
            ▼
    3. Feature extraction                 (src/features.py)
       — title/role gate, skill evidence, experience band,
         company type, location, trajectory, recency, GitHub signal
            │
            ▼
    4. Semantic similarity                (src/scoring.py — EmbeddingIndex)
       — JD vs. pre-computed candidate embeddings (local, offline model)
            │
            ▼
    5. Composite scoring                  (src/scoring.py)
       — weighted feature sum × behavioral availability multiplier
            │
            ▼
    6. Top-100 selection + reasoning      (src/scoring.py, src/reasoning.py)
       — deterministic template, grounded in real sub-scores, no LLM calls
            │
            ▼
    7. CSV output                         (src/rank.py)
```

## Why this design

**No ML/DL model is trained.** There are no labels for the 100K pool — this
is a feature-engineering and scoring-function problem, not a supervised
learning problem. The only deep learning component is a frozen, pre-trained
sentence-transformer model used purely for similarity math, run entirely
offline from local cache.

**Title/role gating exists to defeat keyword-stuffing.** The dataset
contains thousands of off-target candidates (HR Managers, Marketing
Managers, etc.) with AI keywords stuffed into their skills list. A pure
embedding-similarity ranker would rank many of these highly. The title gate
runs before semantic similarity is allowed to dominate the score.

**Honeypot filtering is a hard pre-filter, not a scoring feature.** ~65
candidates are excluded for impossible facts (e.g., "expert" proficiency
with zero months of usage, or years-of-experience claims that don't match
career history by more than 5 years) — directly matching the patterns the
organizers described.

**Behavioral signals are a multiplicative discount, not an additive
feature.** A great-on-paper but inactive candidate gets discounted, never
zeroed out, per the JD's explicit "down-weight appropriately" guidance.

**Reasoning is template-based, not LLM-generated.** Every reasoning string
is built directly from the same computed sub-scores that produced the rank
— guaranteeing no hallucination and structural rank-consistency.

---

## Repository Structure

```
redrob-ranking/
├── README.md                       — this file
├── requirements.txt
├── submission_metadata.yaml        — portal metadata
├── validate_submission.py          — organizer-provided validator
├── conftest.py                     — pytest path config
├── data/
│   ├── candidates.jsonl            — not committed (too large)
│   └── job_description.txt
├── artifacts/
│   ├── candidate_embeddings.npy    — pre-computed (regenerable)
│   └── candidate_ids.npy
├── src/
│   ├── config.py                   — all constants, weights, paths
│   ├── ingest.py                   — Phase 1: streaming JSONL + schema validation
│   ├── honeypot_filter.py          — Phase 2: rule-based exclusion
│   ├── features.py                 — Phase 3: 8 scoring dimensions
│   ├── scoring.py                  — Phase 5: composite score + embedding index
│   ├── reasoning.py                — Phase 6: deterministic reasoning
│   └── rank.py                     — Phase 7: main entrypoint
├── scripts/
│   ├── precompute_embeddings.py    — Phase 4: offline embedding step
│   └── compute_rehearsal.py        — Phase 9: budget compliance check
├── tests/
│   ├── test_ingest.py              — 17 tests
│   └── test_honeypot_filter.py     — 31 tests
└── app/
    └── demo.py                     — Phase 10: Streamlit sandbox demo
```

---

## Testing

```bash
python -m pytest tests/ -v
```

48 tests covering schema validation and every honeypot detection rule,
including the exact patterns named in the job description.

## Compute Budget Compliance

```bash
pip install psutil
python scripts/compute_rehearsal.py
```

Verifies: zero hosted-LLM imports anywhere in `src/`, wall-clock time ≤5
minutes, peak memory ≤16GB.

## Sandbox Demo

```bash
streamlit run app/demo.py
```

Or visit the deployed link in `submission_metadata.yaml`.

---

## Results Summary (measured on full 100K dataset)

| Metric | Value |
|---|---|
| Candidates ingested | 100,000 |
| Schema validation errors | 0 |
| Honeypots excluded | 65 (0.07%) |
| Ranking step wall-clock | 46.7s (budget: 300s) |
| Honeypots in top 100 | 0 |
| Top-10 title-bucket | 100% `core` (genuine ML/AI titles) |