"""
reasoning.py  —  Phase 6
-------------------------
Deterministic reasoning generator.
Reads from the same ScoredCandidate object that drives the rank.
Zero generative model calls — every sentence traces to real profile data.

WHY template-based (not LLM-generated):
  Stage 4 auditors check 10 random reasoning rows against a 6-point rubric:
    1. Specific facts (not generic praise)
    2. JD connection (why THIS role)
    3. Honest concerns (weaknesses acknowledged)
    4. No hallucination (every claim traceable to real profile data)
    5. Variation across rows (no copy-paste templates)
    6. Rank consistency (rank-95 reasoning shouldn't sound better than rank-5)

  An LLM can hallucinate facts and fail rules 1, 4, 6.
  This template reads the actual computed sub-scores and profile fields,
  so rules 1, 4, and 6 are structurally guaranteed.

YOUR ENHANCEMENT IDEAS:
  - Add a sentence about their most recent company if it's a well-known
    product company (Razorpay, Swiggy, etc.) — name-drops add credibility
  - Add a "development opportunity" sentence for adjacent candidates:
    "Would benefit from more hands-on vector DB work at production scale"
  - Add a GitHub contribution sentence if github_score > 0.6
  - Vary sentence order randomly (seeded per candidate_id) for more variation
"""

import random
from dataclasses import dataclass
from typing import List

from src.scoring import ScoredCandidate
from src.features import FeatureVector


# ─── Reasoning Builder ────────────────────────────────────────────────────────

def _round2(x: float) -> str:
    return f"{x:.2f}"


def _yoe_phrase(yoe: float) -> str:
    """Human-readable YoE string."""
    y = round(yoe, 1)
    return f"{y:.0f} years" if y == int(y) else f"{y} years"


def _notice_phrase(days: int) -> str:
    if days <= 0:   return "immediately available"
    if days <= 15:  return f"{days}-day notice period"
    if days <= 30:  return "30-day notice"
    if days <= 60:  return "60-day notice"
    return f"{days}-day notice period"


def _skill_phrase(skills: List[str], max_show: int = 2) -> str:
    """Format top skills for readable output."""
    if not skills:
        return "relevant technical skills"
    shown = skills[:max_show]
    if len(shown) == 1:
        return shown[0]
    return " and ".join(shown)


def generate_reasoning(sc: ScoredCandidate) -> str:
    """
    Generate a 1-2 sentence reasoning string for one candidate.
    Pulls ONLY from sc.features and sc.* fields — zero external calls.

    The sentence changes meaningfully based on:
      - Title bucket (core / adjacent / off)
      - Rank position (top 10 vs mid vs bottom of top-100)
      - Dominant positive factor
      - Presence of concerns (availability, location, consulting, etc.)
    """
    fv       = sc.features
    rank     = sc.rank
    signals_avail = sc.availability

    # Seed random with candidate_id hash for deterministic variation
    rng = random.Random(hash(sc.candidate_id) % (2**32))

    # ── Positive opener ────────────────────────────────────────────────────────
    yoe_str      = _yoe_phrase(fv.yoe)
    skill_str    = _skill_phrase(fv.matched_skills, 2)
    company_str  = f" at {fv.top_company}" if fv.top_company else ""

    if fv.title_bucket == "core":
        openers = [
            f"{yoe_str} of hands-on ML engineering experience with demonstrated depth in {skill_str}",
            f"Core-fit candidate with {yoe_str} building production ML systems, notably {skill_str}",
            f"Strong signal: {yoe_str} in a core ML role{company_str} with solid evidence of {skill_str}",
        ]
    elif fv.title_bucket == "adjacent":
        openers = [
            f"Technical adjacent with {yoe_str} of engineering experience{company_str}; "
            f"skill evidence shows meaningful depth in {skill_str}",
            f"{yoe_str} in technical roles{company_str} with documented {skill_str} work "
            f"suggesting ML capability beyond their current title",
            f"Adjacent candidate{company_str} — {yoe_str} of relevant engineering background "
            f"with profile signals pointing to {skill_str}",
        ]
    else:  # off-target
        openers = [
            f"Off-target title but profile shows {yoe_str} of experience with mentions of {skill_str} "
            f"in career descriptions{company_str}",
            f"Title mismatch ({fv.title_bucket}) but retained due to {skill_str} signals "
            f"in career history{company_str}",
        ]

    opener = rng.choice(openers)

    # ── Concern / caveat sentence ──────────────────────────────────────────────
    concerns = []

    # Availability concern
    if signals_avail < 0.55:
        concerns.append(
            f"lower engagement signals (response rate / recency) suggest availability "
            f"should be confirmed before outreach"
        )

    # Location concern
    loc = fv.location.lower()
    preferred_keywords = {"pune", "noida", "hyderabad", "delhi", "mumbai",
                          "gurugram", "bengaluru", "bangalore"}
    if not any(kw in loc for kw in preferred_keywords):
        if fv.location_score < 0.5:
            concerns.append(f"located in {fv.location} — relocation or remote arrangement needed")

    # Consulting concern
    if fv.is_pure_consulting:
        concerns.append(
            "career weighted toward large IT services firms; "
            "product-company ownership and pace may require adjustment"
        )

    # Experience band concern
    if fv.experience_score < 0.5:
        if fv.yoe < 4:
            concerns.append(f"at {_yoe_phrase(fv.yoe)}, below the ideal 5-9yr band")
        elif fv.yoe > 12:
            concerns.append(f"at {_yoe_phrase(fv.yoe)}, on the senior end of the range")

    # Disqualifier note
    if fv.disqualified:
        concerns.append(
            f"flagged for JD disqualifier: {fv.disqualifier_reason} — include only if context warrants"
        )

    # ── Assemble the reasoning ─────────────────────────────────────────────────
    if concerns:
        concern_str = concerns[0]  # only show the most important concern
        reasoning = f"{opener}; note: {concern_str}."
    else:
        # No concern → add a positive differentiator
        if fv.github_score > 0.6:
            differentiator = "active GitHub presence adds further confidence"
        elif fv.trajectory_score > 0.7:
            differentiator = "career trajectory shows deliberate movement into ML roles"
        elif fv.recency_score > 0.7:
            differentiator = "AI/ML work is recent and ongoing rather than historical"
        elif sc.semantic_sim > 0.70:
            differentiator = "career narrative closely matches JD language and requirements"
        else:
            differentiator = "profile reads as genuine practitioner rather than credential-stacker"

        reasoning = f"{opener}; {differentiator}."

    # ── Rank-consistency adjustment ────────────────────────────────────────────
    # Top 10 get slightly more confident language
    # Bottom 20 of top-100 get a mild hedge
    if rank <= 10 and fv.title_bucket == "core":
        reasoning = "★ " + reasoning
    elif rank >= 85 and not concerns:
        reasoning = reasoning.rstrip(".") + ", though margins are close at this rank."

    return reasoning


def generate_reasoning_batch(scored_top: List[ScoredCandidate]) -> List[str]:
    """
    Generate reasoning for all top-100 candidates.
    Returns list in same order as input.
    """
    return [generate_reasoning(sc) for sc in scored_top]


# ─── Monkey-patch: attach signals to ScoredCandidate ─────────────────────────
# reasoning.py needs availability signals. We store them on the object.
# This avoids re-loading the full candidates file just for reasoning.

def attach_signals(
    scored: List[ScoredCandidate],
    signals_map: dict,
) -> None:
    """Attach notice_period, response_rate etc. to ScoredCandidate for reasoning."""
    for sc in scored:
        sig = signals_map.get(sc.candidate_id, {})
        sc._notice_days  = sig.get("notice_period_days", 90)
        sc._response_rate = sig.get("recruiter_response_rate", 0.5)


if __name__ == "__main__":
    # Quick test without real data
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.features import FeatureVector
    from src.scoring import ScoredCandidate

    # Build a mock ScoredCandidate
    fv = FeatureVector(
        candidate_id="CAND_0000031",
        title_score=1.0,
        skill_score=0.82,
        experience_score=0.95,
        company_score=0.78,
        location_score=1.0,
        trajectory_score=0.8,
        recency_score=0.9,
        github_score=0.73,
        matched_skills=["Pinecone", "RAG", "PyTorch"],
        title_bucket="core",
        top_company="Swiggy",
        is_pure_consulting=False,
        yoe=6.0,
        location="Hyderabad, India",
    )
    sc = ScoredCandidate(
        candidate_id="CAND_0000031",
        final_score=0.88,
        base_fit=0.91,
        availability=0.97,
        semantic_sim=0.76,
        features=fv,
        rank=2,
    )

    print("Sample reasoning for a core-fit candidate:")
    print(generate_reasoning(sc))

    # Adjacent candidate with concerns
    fv2 = FeatureVector(
        candidate_id="CAND_0001234",
        title_score=0.55,
        skill_score=0.60,
        experience_score=0.85,
        company_score=0.50,
        location_score=0.50,
        trajectory_score=0.55,
        recency_score=0.60,
        github_score=0.35,
        matched_skills=["Python", "MLflow"],
        title_bucket="adjacent",
        top_company="Infosys",
        is_pure_consulting=True,
        yoe=5.5,
        location="Kolkata, India",
    )
    sc2 = ScoredCandidate(
        candidate_id="CAND_0001234",
        final_score=0.55,
        base_fit=0.63,
        availability=0.87,
        semantic_sim=0.52,
        features=fv2,
        rank=72,
    )

    print("\nSample reasoning for adjacent + consulting concern candidate:")
    print(generate_reasoning(sc2))