"""
config.py
---------
Single source of truth for ALL constants, weights, paths, and
JD-derived requirements. Every other module imports from here.
Change weights here only — nowhere else.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent.parent
DATA_DIR        = ROOT_DIR / "data"
ARTIFACTS_DIR   = ROOT_DIR / "artifacts"
SRC_DIR         = ROOT_DIR / "src"

CANDIDATES_FILE     = DATA_DIR / "candidates.jsonl"
EMBEDDINGS_FILE     = ARTIFACTS_DIR / "candidate_embeddings.npy"
CANDIDATE_IDS_FILE  = ARTIFACTS_DIR / "candidate_ids.npy"
JD_FILE             = ROOT_DIR / "data" / "job_description.txt"

# ─── Embedding Model ──────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: 384-dim, ~80MB, CPU-fast, no network needed at runtime
EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE = 512          # tune down if RAM is tight
EMBEDDING_DIM       = 384

# ─── Submission Constants ─────────────────────────────────────────────────────
TOP_K               = 100           # exactly 100 candidates in output
SUBMISSION_COLUMNS  = ["candidate_id", "rank", "score", "reasoning"]
SCORE_DECIMALS      = 4             # round final scores to 4dp

# ─── JD-Derived: Experience Band ─────────────────────────────────────────────
# JD says "5-9 years" is the sweet spot; it's a soft range, not hard cutoff
YOE_MIN_IDEAL       = 5
YOE_MAX_IDEAL       = 9
YOE_ABSOLUTE_MIN    = 2            # below this: steep penalty
YOE_ABSOLUTE_MAX    = 18           # above this: mild penalty (over-qualified)

# ─── JD-Derived: Title Classification ────────────────────────────────────────
# These come from the ACTUAL title distribution we found in the 100K dataset.
# core_fit   → direct match for the role (score multiplier = 1.0)
# adjacent   → technical but not ML-core (score multiplier = 0.55)
# off_target → clearly wrong domain (score multiplier = 0.05)

CORE_FIT_TITLES = {
    "AI Engineer",
    "ML Engineer",
    "Machine Learning Engineer",
    "Senior Machine Learning Engineer",
    "Staff Machine Learning Engineer",
    "Lead AI Engineer",
    "Senior AI Engineer",
    "AI Research Engineer",
    "Applied ML Engineer",
    "Data Scientist",
    "Senior Data Scientist",
    "NLP Engineer",
    "Senior NLP Engineer",
    "Computer Vision Engineer",
    "Senior Software Engineer (ML)",
    "Recommendation Systems Engineer",
    "Search Engineer",
    "AI Specialist",
    "Junior ML Engineer",
    "Senior Applied Scientist",
}

ADJACENT_TITLES = {
    "Software Engineer",
    "Senior Software Engineer",
    "Backend Engineer",
    "Full Stack Developer",
    "Cloud Engineer",
    "DevOps Engineer",
    "Data Engineer",
    "Senior Data Engineer",
    "Analytics Engineer",
    "Data Analyst",
    "Frontend Engineer",
    "Mobile Developer",
    "Java Developer",
    ".NET Developer",
    "QA Engineer",
}

# Everything NOT in core or adjacent → off_target (HR Manager, Accountant, etc.)

TITLE_MULTIPLIERS = {
    "core":     1.0,
    "adjacent": 0.55,
    "off":      0.05,
}

# ─── JD-Derived: Consulting Company Penalty ───────────────────────────────────
# JD explicitly says: "Pure consulting career → penalize.
#  But if they have prior product-company experience, that's fine."
CONSULTING_COMPANIES = {
    "tcs", "tata consultancy services",
    "infosys",
    "wipro",
    "accenture",
    "cognizant",
    "capgemini",
    "hcl", "hcl technologies",
    "tech mahindra",
    "mphasis",
    "hexaware",
    "l&t infotech", "ltimindtree",
}

CONSULTING_PENALTY   = 0.6   # multiplier when entire career is pure consulting
# No penalty if any non-consulting role exists in career history

# ─── JD-Derived: Preferred Locations ─────────────────────────────────────────
PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai",
    "delhi", "gurugram", "gurgaon", "bengaluru", "bangalore",
    "chennai", "delhi ncr",
}

LOCATION_SCORE_IN_PREFERRED    = 1.0
LOCATION_SCORE_WILLING_RELOCATE = 0.7
LOCATION_SCORE_INDIA_OTHER     = 0.5
LOCATION_SCORE_INTERNATIONAL   = 0.3

# ─── JD-Derived: Critical Skills (with evidence weighting) ───────────────────
# Tier 1: Must-have (high weight)
SKILLS_TIER1 = {
    "python", "machine learning", "deep learning", "nlp",
    "embeddings", "semantic search", "rag",
    "vector database", "faiss", "pinecone", "qdrant", "milvus", "weaviate",
    "llm", "llms", "transformers", "hugging face",
    "pytorch", "tensorflow",
    "mlops", "mlflow", "bentoml",
    "fastapi",
}

# Tier 2: Strong signals (medium weight)
SKILLS_TIER2 = {
    "langchain", "lora", "qlora", "fine-tuning",
    "docker", "kubernetes", "aws", "gcp", "azure",
    "elasticsearch", "redis",
    "spark", "airflow",
    "scikit-learn", "xgboost",
    "openai api", "anthropic api",
}

# Tier 3: Nice-to-have
SKILLS_TIER3 = {
    "sql", "postgresql", "mongodb",
    "kafka", "dbt",
    "onnx", "tts", "speech recognition",
    "opencv", "gans", "diffusion models", "yolo",
    "git", "ci/cd", "github actions",
}

SKILL_WEIGHTS = {"tier1": 1.0, "tier2": 0.55, "tier3": 0.25}

# Proficiency multipliers (applied ON TOP of skill weight)
PROFICIENCY_WEIGHTS = {
    "expert":       1.0,
    "advanced":     0.75,
    "intermediate": 0.45,
    "beginner":     0.15,
}

# Duration scaling: 12+ months → full credit, scales down linearly below
SKILL_DURATION_FULL_CREDIT_MONTHS = 12

# ─── Composite Score Weights ──────────────────────────────────────────────────
# These weights sum to 1.0. Title gate and skill evidence are heaviest
# because they are the primary keyword-stuffer trap defenses.
WEIGHTS = {
    "title":       0.30,   # title/role coherence gate
    "skill":       0.28,   # evidence-weighted skill score
    "semantic":    0.20,   # embedding cosine similarity to JD
    "experience":  0.10,   # experience band fit
    "company":     0.07,   # company type (product vs consulting)
    "location":    0.05,   # location + relocation signal
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# ─── Behavioral Availability Multiplier ──────────────────────────────────────
# Applied MULTIPLICATIVELY: final_score = base_fit * availability_multiplier
# Range: 0.40 → 1.0  (never zero — don't erase a genuinely skilled candidate)
AVAILABILITY_MIN    = 0.40
AVAILABILITY_MAX    = 1.0

# How long before "last_active" is considered stale (in days)
ACTIVE_RECENT_DAYS  = 60    # < 60 days → full credit
ACTIVE_STALE_DAYS   = 180   # > 180 days → significant discount

# ─── Honeypot Detection Thresholds ───────────────────────────────────────────
# Need 2+ rule violations to HARD-EXCLUDE (1 violation = flagged/logged only)
HONEYPOT_YOE_MISMATCH_THRESHOLD_YEARS = 5   # stated vs career-history total
HONEYPOT_MIN_FLAGS_FOR_EXCLUSION       = 1   # ≥1 flag → exclude (conservative)

# ─── GitHub Signal ────────────────────────────────────────────────────────────
GITHUB_NO_ACCOUNT   = -1    # sentinel value meaning "no GitHub linked"