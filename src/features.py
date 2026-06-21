"""
features.py  —  Phase 3
------------------------
Six independent, deterministic scoring dimensions.
These are the core anti-keyword-stuffing engine.

WHY each dimension exists (tied to JD + organizer rules):
  1. title_score      → kills keyword-stuffers by checking WHO the person IS, not what they list
  2. skill_score      → weights by evidence (duration, endorsements), not just skill presence
  3. experience_score → soft-bands the 5-9yr JD range, hard-floors the disqualifiers
  4. company_score    → penalizes pure-consulting careers per explicit JD rule
  5. location_score   → Pune/Noida preference from JD
  6. career_text_score→ semantic quality signal from career descriptions (pre-embedding hook)

YOUR IDEAS TO ENHANCE THIS FILE:
  - Add a "trajectory score": is the candidate moving UP toward ML roles or sideways?
  - Add a "product company ratio": what fraction of career was at product vs service companies?
  - Add a "recency of AI work score": did they do AI work in the last 2 years specifically?
  - Add a "open source contribution score" from github_activity_score signal
  - Add a "education tier bonus" for IITs/NITs/top universities
  These would all go into the FeatureVector dataclass below and get weighted in scoring.py
"""

import re
import logging
from datetime import date, datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

from src.config import (
    CORE_FIT_TITLES, ADJACENT_TITLES, TITLE_MULTIPLIERS,
    CONSULTING_COMPANIES, CONSULTING_PENALTY,
    PREFERRED_LOCATIONS,
    LOCATION_SCORE_IN_PREFERRED, LOCATION_SCORE_WILLING_RELOCATE,
    LOCATION_SCORE_INDIA_OTHER, LOCATION_SCORE_INTERNATIONAL,
    SKILLS_TIER1, SKILLS_TIER2, SKILLS_TIER3, SKILL_WEIGHTS,
    PROFICIENCY_WEIGHTS, SKILL_DURATION_FULL_CREDIT_MONTHS,
    YOE_MIN_IDEAL, YOE_MAX_IDEAL, YOE_ABSOLUTE_MIN, YOE_ABSOLUTE_MAX,
)

logger = logging.getLogger(__name__)
TODAY = date(2026, 6, 18)


# ─── Output: Feature Vector ───────────────────────────────────────────────────
# This dataclass is the contract between features.py and scoring.py.
# Every field is a float in [0, 1]. Reasoning generation reads from this too.

@dataclass
class FeatureVector:
    candidate_id:       str   = ""

    # Core scoring dimensions (each 0.0 → 1.0)
    title_score:        float = 0.0   # role/domain coherence
    skill_score:        float = 0.0   # evidence-weighted skill match
    experience_score:   float = 0.0   # YoE band fit + disqualifier check
    company_score:      float = 0.0   # product vs consulting career
    location_score:     float = 0.0   # geo fit + relocation signal

    # Enhancement dimensions (your ideas plugged in here)
    trajectory_score:   float = 0.0   # career moving TOWARD ML roles?
    recency_score:      float = 0.0   # AI work in last 2 years specifically?
    github_score:       float = 0.0   # open-source / coding activity

    # Metadata for reasoning generation
    matched_skills:     List[str] = field(default_factory=list)  # top matching skills with evidence
    title_bucket:       str = "off"   # core / adjacent / off
    top_company:        str = ""      # most recent company name
    is_pure_consulting: bool = False  # flag for reasoning
    yoe:                float = 0.0   # raw years for reasoning string
    location:           str = ""      # raw location for reasoning
    disqualified:       bool = False  # hard JD disqualifier hit
    disqualifier_reason: str = ""     # what triggered it


# ─── 1. TITLE / ROLE COHERENCE SCORE ─────────────────────────────────────────
# PURPOSE: Kill the keyword-stuffing trap at the root.
#          A Marketing Manager is "off" even with perfect skills.
#
# YOUR ENHANCEMENT IDEA:
#   Instead of just current_title, also look at the MOST RECENT career_history
#   title — someone who was an ML Engineer last month but is now "Job Seeker"
#   shouldn't lose their core-fit status.

def compute_title_score(candidate: Dict[str, Any]) -> Tuple[float, str, str]:
    """
    Returns (score, bucket, matched_title).
    bucket ∈ {'core', 'adjacent', 'off'}
    """
    current_title = candidate["profile"]["current_title"].strip()

    # Check current title first
    if current_title in CORE_FIT_TITLES:
        return TITLE_MULTIPLIERS["core"], "core", current_title

    if current_title in ADJACENT_TITLES:
        # Enhancement: check if ANY recent career role is core-fit
        # (last 2 roles — catches recently transitioned candidates)
        recent_titles = [
            h["title"] for h in candidate.get("career_history", [])[:2]
        ]
        if any(t in CORE_FIT_TITLES for t in recent_titles):
            # Bonus: adjacent title but was recently doing ML work
            return 0.72, "adjacent", current_title
        return TITLE_MULTIPLIERS["adjacent"], "adjacent", current_title

    # Not in either set → check career history for RECENT core-fit roles
    # (someone who transitioned to "Project Manager" from ML in last 1 year)
    recent_core = [
        h for h in candidate.get("career_history", [])[:2]
        if h["title"] in CORE_FIT_TITLES
    ]
    if recent_core:
        # Discounted: recent ML background but current title isn't ML
        return 0.45, "adjacent", current_title

    return TITLE_MULTIPLIERS["off"], "off", current_title


# ─── 2. SKILL EVIDENCE SCORE ─────────────────────────────────────────────────
# PURPOSE: Weight skills by EVIDENCE (duration × proficiency × endorsements),
#          not just presence. Expert skill with 0 months = near zero.
#
# YOUR ENHANCEMENT IDEA:
#   Add a "skill cluster" bonus: if a candidate has MULTIPLE related skills
#   (e.g. Pinecone + FAISS + Qdrant = "vector DB expert cluster"),
#   award a cluster bonus because this shows depth, not breadth sampling.

VECTOR_DB_CLUSTER   = {"pinecone", "qdrant", "milvus", "weaviate", "faiss",
                        "elasticsearch", "opensearch", "chroma", "pgvector"}
LLM_CLUSTER         = {"rag", "langchain", "llm", "llms", "fine-tuning", "lora",
                        "qlora", "peft", "hugging face", "hugging face transformers",
                        "openai api", "anthropic api", "embeddings", "semantic search"}
MLOPS_CLUSTER       = {"mlflow", "bentoml", "kubeflow", "airflow", "mlops",
                        "weights & biases", "wandb", "docker", "kubernetes"}

def _duration_factor(duration_months: int) -> float:
    """Scale duration to [0, 1]. Full credit at 12+ months."""
    if duration_months >= SKILL_DURATION_FULL_CREDIT_MONTHS:
        return 1.0
    if duration_months <= 0:
        return 0.02  # near-zero but not absolute zero (allows single-project work)
    return duration_months / SKILL_DURATION_FULL_CREDIT_MONTHS


def _endorsement_factor(endorsements: int) -> float:
    """Endorsements boost: diminishing returns above 20."""
    if endorsements <= 0:
        return 0.85   # no endorsements doesn't mean bad, just unverified
    return min(1.0, 0.85 + (endorsements / 100) * 0.15)


def compute_skill_score(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Returns (score, matched_skills_list_for_reasoning).
    matched_skills contains the top-3 skills with real evidence, named exactly.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return 0.0, []

    raw_score   = 0.0
    max_possible = 0.0
    skill_details: List[Tuple[float, str]] = []  # (contribution, name)

    # Cluster tracking
    vdb_hits, llm_hits, mlops_hits = 0, 0, 0

    for s in skills:
        name_lower = s["name"].lower()
        prof       = s.get("proficiency", "beginner")
        duration   = s.get("duration_months", 0)
        endorsements = s.get("endorsements", 0)

        # Tier classification
        if name_lower in SKILLS_TIER1:
            tier_weight = SKILL_WEIGHTS["tier1"]
        elif name_lower in SKILLS_TIER2:
            tier_weight = SKILL_WEIGHTS["tier2"]
        elif name_lower in SKILLS_TIER3:
            tier_weight = SKILL_WEIGHTS["tier3"]
        else:
            continue  # skill not JD-relevant → skip

        # Evidence-weighted contribution
        prof_factor     = PROFICIENCY_WEIGHTS.get(prof, 0.15)
        dur_factor      = _duration_factor(duration)
        endorse_factor  = _endorsement_factor(endorsements)

        contribution = tier_weight * prof_factor * dur_factor * endorse_factor
        raw_score   += contribution
        max_possible += tier_weight  # max if all factors = 1.0

        skill_details.append((contribution, s["name"]))

        # Track cluster membership
        if name_lower in VECTOR_DB_CLUSTER: vdb_hits += 1
        if name_lower in LLM_CLUSTER:       llm_hits += 1
        if name_lower in MLOPS_CLUSTER:     mlops_hits += 1

    # Cluster bonus (enhancement idea implemented)
    cluster_bonus = 0.0
    if vdb_hits >= 2:   cluster_bonus += 0.06  # "Vector DB expert" cluster
    if llm_hits >= 3:   cluster_bonus += 0.06  # "LLM practitioner" cluster
    if mlops_hits >= 2: cluster_bonus += 0.04  # "MLOps" cluster

    # Normalize
    if max_possible > 0:
        normalized = min(1.0, (raw_score / max_possible) + cluster_bonus)
    else:
        normalized = 0.0

    # Top skills by contribution for reasoning
    skill_details.sort(key=lambda x: x[0], reverse=True)
    top_skills = [name for _, name in skill_details[:4] if _ > 0.05]

    return round(normalized, 4), top_skills


# ─── 3. EXPERIENCE BAND SCORE ────────────────────────────────────────────────
# PURPOSE: Soft-band 5-9yr JD range. Hard-floor explicit disqualifiers.
#
# DISQUALIFIERS from JD (hard, not soft):
#   - Pure research, no production deployment
#   - AI experience = only recent LangChain (< 12 months)
#   - Hasn't written production code in 18+ months (senior architect types)
#
# YOUR ENHANCEMENT IDEA:
#   "Career velocity" — how long did it take them to reach ML roles?
#   Someone who went from zero to ML Engineer in 3 years is more impressive
#   than someone who drifted into ML after 12 years of Java.

def _infer_disqualifier(candidate: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check JD's hard disqualifiers from career_history patterns.
    Returns (is_disqualified, reason).
    """
    history = candidate.get("career_history", [])
    if not history:
        return False, ""

    titles_lower = [h["title"].lower() for h in history]
    desc_combined = " ".join(h.get("description", "") for h in history).lower()

    # Disqualifier: Every role is "Architect" / "Tech Lead" with no IC title
    # Proxy for "hasn't written production code in 18+ months"
    mgmt_keywords = {"architect", "tech lead", "engineering manager",
                     "vp of engineering", "cto", "director of engineering"}
    ic_keywords   = {"engineer", "scientist", "developer", "analyst", "researcher"}
    all_mgmt = all(any(k in t for k in mgmt_keywords) for t in titles_lower)
    has_ic   = any(any(k in t for k in ic_keywords) for t in titles_lower)
    if all_mgmt and not has_ic and len(titles_lower) >= 2:
        return True, "pure_management_no_IC_roles"

    # Disqualifier: Only very recent AI experience with no pre-LLM ML history
    # Detect: no AI keywords in desc of roles older than 12 months, only recent ones
    ai_keywords = {"llm", "embedding", "vector", "rag", "transformer",
                   "pytorch", "machine learning", "deep learning", "nlp"}

    has_pre_llm_ml = False
    for h in history:
        try:
            end = (datetime.strptime(h["end_date"], "%Y-%m-%d").date()
                   if h.get("end_date") else TODAY)
            months_ago = (TODAY - end).days / 30
            if months_ago > 12:  # older than 12 months
                desc = h.get("description", "").lower()
                if any(k in desc for k in ai_keywords):
                    has_pre_llm_ml = True
                    break
        except (ValueError, KeyError):
            pass

    return False, ""


def compute_experience_score(candidate: Dict[str, Any]) -> Tuple[float, float, bool, str]:
    """
    Returns (score, yoe, is_disqualified, disqualifier_reason).
    """
    yoe = candidate["profile"].get("years_of_experience", 0)

    # Hard disqualifier check
    is_dq, dq_reason = _infer_disqualifier(candidate)
    if is_dq:
        return 0.05, yoe, True, dq_reason

    # Soft band scoring
    if YOE_MIN_IDEAL <= yoe <= YOE_MAX_IDEAL:
        score = 1.0                                    # perfect band
    elif yoe < YOE_MIN_IDEAL:
        if yoe < YOE_ABSOLUTE_MIN:
            score = 0.2                                # too junior
        else:
            score = 0.55 + 0.45 * (yoe - YOE_ABSOLUTE_MIN) / (YOE_MIN_IDEAL - YOE_ABSOLUTE_MIN)
    else:  # yoe > YOE_MAX_IDEAL
        if yoe > YOE_ABSOLUTE_MAX:
            score = 0.5                                # significantly over-qualified
        else:
            score = 1.0 - 0.5 * (yoe - YOE_MAX_IDEAL) / (YOE_ABSOLUTE_MAX - YOE_MAX_IDEAL)

    return round(score, 4), yoe, False, ""


# ─── 4. COMPANY TYPE SCORE ────────────────────────────────────────────────────
# PURPOSE: JD explicitly says pure-consulting career = penalized.
#          BUT if they have even ONE product-company role, no penalty.
#
# YOUR ENHANCEMENT IDEA:
#   "Product company ratio" = (months at product companies) / (total career months)
#   A nuanced 0-1 score instead of binary penalty.
#   Also: award bonus for FAANG/top Indian product companies (Razorpay, CRED, etc.)

PREMIUM_COMPANIES = {
    "google", "microsoft", "meta", "amazon", "apple",
    "openai", "anthropic", "deepmind", "nvidia",
    "razorpay", "cred", "swiggy", "zomato", "flipkart",
    "meesho", "dream11", "phonepe", "paytm", "freshworks",
    "ola", "byju's", "unacademy", "zepto",
}

def compute_company_score(candidate: Dict[str, Any]) -> Tuple[float, bool, str]:
    """
    Returns (score, is_pure_consulting, most_recent_notable_company).
    """
    history  = candidate.get("career_history", [])
    if not history:
        return 0.5, False, ""

    total_months       = 0
    consulting_months  = 0
    product_months     = 0
    premium_months     = 0
    most_recent_co     = history[0].get("company", "") if history else ""

    for h in history:
        co_lower = h.get("company", "").lower().strip()
        months   = h.get("duration_months", 0)
        total_months += months

        if any(c in co_lower for c in CONSULTING_COMPANIES):
            consulting_months += months
        else:
            product_months += months
            if any(p in co_lower for p in PREMIUM_COMPANIES):
                premium_months += months

    if total_months == 0:
        return 0.5, False, most_recent_co

    consulting_ratio = consulting_months / total_months
    product_ratio    = product_months    / total_months
    premium_ratio    = premium_months    / total_months

    is_pure_consulting = (consulting_ratio > 0.85)

    # Base score: product company ratio
    base = 0.4 + 0.6 * product_ratio

    # Premium bonus (up to +0.15)
    base += 0.15 * min(1.0, premium_ratio * 3)

    # Apply consulting penalty if pure consulting
    if is_pure_consulting:
        base *= CONSULTING_PENALTY

    return round(min(1.0, base), 4), is_pure_consulting, most_recent_co


# ─── 5. LOCATION SCORE ────────────────────────────────────────────────────────
# PURPOSE: JD prefers Pune/Noida/Hyderabad/Delhi NCR.
#          Willing-to-relocate partially compensates for being elsewhere.

def compute_location_score(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Returns (score, location_string_for_reasoning).
    """
    location = candidate["profile"].get("location", "").lower()
    country  = candidate["profile"].get("country",  "").lower()
    relocate = candidate["redrob_signals"].get("willing_to_relocate", False)

    location_str = candidate["profile"].get("location", "Unknown")

    if any(loc in location for loc in PREFERRED_LOCATIONS):
        return LOCATION_SCORE_IN_PREFERRED, location_str

    if relocate and country in {"india", "in"}:
        return LOCATION_SCORE_WILLING_RELOCATE, location_str

    if country in {"india", "in"}:
        return LOCATION_SCORE_INDIA_OTHER, location_str

    return LOCATION_SCORE_INTERNATIONAL, location_str


# ─── 6. TRAJECTORY SCORE (Enhancement idea implemented) ──────────────────────
# PURPOSE: Is the candidate moving TOWARD ML/AI roles over time?
#          A civil engineer who pivoted to ML 2 years ago is MORE interesting
#          than someone who has been stagnant in a non-ML title for 5 years.
#
# HOW IT WORKS:
#   Sort career history by date → check if recent roles are higher-tier
#   than older roles. Upward trajectory = bonus.

def compute_trajectory_score(candidate: Dict[str, Any]) -> float:
    """
    Returns a score in [0, 1] reflecting career movement TOWARD ML roles.
    0.5 = neutral/stable. >0.5 = positive trajectory. <0.5 = stagnant or declining.
    """
    history = candidate.get("career_history", [])
    if len(history) < 2:
        return 0.5  # not enough data

    def title_tier(title: str) -> int:
        if title in CORE_FIT_TITLES:   return 3
        if title in ADJACENT_TITLES:   return 2
        return 1  # off-target

    # Sort by start_date to get chronological order
    try:
        sorted_hist = sorted(
            history,
            key=lambda h: datetime.strptime(h["start_date"], "%Y-%m-%d").date()
        )
    except (ValueError, KeyError):
        return 0.5

    early_tier = title_tier(sorted_hist[0]["title"])
    recent_tier = title_tier(sorted_hist[-1]["title"])

    if recent_tier > early_tier:
        return 0.80   # positive: moved UP toward ML
    if recent_tier == early_tier:
        return 0.55   # stable
    return 0.35       # declining (was ML, now in a less ML role)


# ─── 7. RECENCY OF AI WORK (Enhancement idea implemented) ────────────────────
# PURPOSE: Did the candidate do real AI/ML work in the LAST 24 MONTHS?
#          This catches stale AI credentials — someone who did ML in 2019
#          but has been doing Java since 2022 should score lower.

AI_WORK_KEYWORDS = {
    "embedding", "vector", "rag", "llm", "transformer", "fine-tun",
    "pytorch", "tensorflow", "machine learning", "deep learning",
    "nlp", "recommendation", "ranking", "retrieval", "semantic",
    "hugging face", "langchain", "pinecone", "faiss", "qdrant",
    "model train", "model deploy", "inference", "mlops", "mlflow",
}

def compute_recency_score(candidate: Dict[str, Any]) -> float:
    """
    Returns 0-1: how recently the candidate did real AI/ML work.
    Based on career_history descriptions from the last 24 months.
    """
    history = candidate.get("career_history", [])
    if not history:
        return 0.1

    recent_ai_months  = 0
    total_recent_months = 0

    for h in history:
        try:
            start = datetime.strptime(h["start_date"], "%Y-%m-%d").date()
            end   = (datetime.strptime(h["end_date"], "%Y-%m-%d").date()
                     if h.get("end_date") else TODAY)
            months_ago_start = (TODAY - start).days / 30
            months_ago_end   = (TODAY - end).days / 30

            if months_ago_end > 24:
                continue  # role ended more than 2 years ago → skip

            overlap_months = h.get("duration_months", 0)
            total_recent_months += overlap_months

            desc = h.get("description", "").lower()
            ai_hits = sum(1 for kw in AI_WORK_KEYWORDS if kw in desc)

            if ai_hits >= 2:
                recent_ai_months += overlap_months
            elif ai_hits == 1:
                recent_ai_months += overlap_months * 0.5

        except (ValueError, KeyError):
            pass

    if total_recent_months == 0:
        return 0.3  # no recent career data at all

    ratio = recent_ai_months / total_recent_months
    return round(min(1.0, ratio + 0.2), 4)  # +0.2 floor bonus for having recent roles


# ─── 8. GITHUB / OPEN-SOURCE SCORE (Enhancement idea implemented) ────────────
# PURPOSE: github_activity_score from redrob_signals.
#          -1 means no GitHub linked. 0-100 = activity score.
#          JD values open-source contributions explicitly.

def compute_github_score(candidate: Dict[str, Any]) -> float:
    """
    Returns 0-1 normalized GitHub activity.
    -1 (no account) → small penalty vs neutral, not zero.
    """
    raw = candidate["redrob_signals"].get("github_activity_score", -1)
    if raw == -1:
        return 0.35   # no GitHub: small discount, not zero
    return round(raw / 100.0, 4)


# ─── MAIN: Compute all features for one candidate ────────────────────────────

def compute_features(candidate: Dict[str, Any]) -> FeatureVector:
    """
    Runs all feature extractors on one candidate.
    Returns a FeatureVector with all scores filled in.
    This is what scoring.py consumes.
    """
    cid = candidate["candidate_id"]

    # 1. Title
    t_score, t_bucket, _ = compute_title_score(candidate)

    # 2. Skills
    sk_score, matched_skills = compute_skill_score(candidate)

    # 3. Experience
    exp_score, yoe, is_dq, dq_reason = compute_experience_score(candidate)

    # 4. Company
    co_score, is_consulting, top_company = compute_company_score(candidate)

    # 5. Location
    loc_score, location_str = compute_location_score(candidate)

    # 6. Trajectory (enhancement)
    traj_score = compute_trajectory_score(candidate)

    # 7. Recency (enhancement)
    rec_score = compute_recency_score(candidate)

    # 8. GitHub (enhancement)
    gh_score = compute_github_score(candidate)

    return FeatureVector(
        candidate_id        = cid,
        title_score         = t_score,
        skill_score         = sk_score,
        experience_score    = exp_score,
        company_score       = co_score,
        location_score      = loc_score,
        trajectory_score    = traj_score,
        recency_score       = rec_score,
        github_score        = gh_score,
        matched_skills      = matched_skills,
        title_bucket        = t_bucket,
        top_company         = top_company,
        is_pure_consulting  = is_consulting,
        yoe                 = yoe,
        location            = location_str,
        disqualified        = is_dq,
        disqualifier_reason = dq_reason,
    )


def compute_features_batch(
    candidates: List[Dict[str, Any]]
) -> List[FeatureVector]:
    """
    Compute features for a list of candidates.
    Returns list of FeatureVectors in same order.
    """
    results = []
    for c in candidates:
        try:
            fv = compute_features(c)
        except Exception as e:
            logger.warning(f"{c.get('candidate_id','?')}: feature error — {e}")
            fv = FeatureVector(candidate_id=c.get("candidate_id", "?"))
        results.append(fv)
    return results


if __name__ == "__main__":
    import sys, time
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from src.ingest import load_all_candidates
    from src.honeypot_filter import filter_candidates

    data_path = __import__("pathlib").Path("data/candidates.jsonl")
    print("Loading 2000 candidates for feature smoke test...")
    candidates = load_all_candidates(data_path, validate=True, max_records=2000)
    clean, _ = filter_candidates(candidates)

    t0 = time.time()
    fvecs = compute_features_batch(clean)
    elapsed = time.time() - t0

    # Show top scorers by skill + title
    from dataclasses import asdict
    top = sorted(fvecs, key=lambda x: x.title_score * 0.4 + x.skill_score * 0.6, reverse=True)[:10]
    print(f"\n✅ Features computed in {elapsed:.1f}s for {len(fvecs)} candidates")
    print("\nTop 10 by title+skill (sanity check — should be ML/AI titles):")
    for fv in top:
        print(f"  {fv.candidate_id} | {fv.title_bucket:8s} | "
              f"title={fv.title_score:.2f} skill={fv.skill_score:.2f} "
              f"exp={fv.experience_score:.2f} traj={fv.trajectory_score:.2f} "
              f"| skills={fv.matched_skills[:3]}")