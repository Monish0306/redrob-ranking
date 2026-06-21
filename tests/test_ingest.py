"""
tests/test_ingest.py
Unit tests for the ingestion and schema validation layer.
Run: pytest tests/test_ingest.py -v
"""

import json
import pytest
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ingest import validate_candidate, stream_candidates, load_all_candidates


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def make_valid_candidate(candidate_id="CAND_0000001"):
    """Return a minimal, fully valid candidate record."""
    return {
        "candidate_id": candidate_id,
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer at Acme",
            "summary": "5 years building production ML systems.",
            "location": "Pune, MH",
            "country": "India",
            "years_of_experience": 6.0,
            "current_title": "ML Engineer",
            "current_company": "Acme AI",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "Acme AI",
                "title": "ML Engineer",
                "start_date": "2020-01-01",
                "end_date": None,
                "duration_months": 66,
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Built RAG pipelines and fine-tuned LLMs.",
            }
        ],
        "education": [],
        "skills": [
            {"name": "Python",  "proficiency": "expert",       "endorsements": 20, "duration_months": 60},
            {"name": "PyTorch", "proficiency": "advanced",     "endorsements": 10, "duration_months": 36},
        ],
        "certifications": [],
        "languages": [],
        "redrob_signals": {
            "profile_completeness_score": 88,
            "signup_date": "2020-01-15",
            "last_active_date": "2026-06-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 5,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 4.0,
            "skill_assessment_scores": {"Python": 90},
            "connection_count": 200,
            "endorsements_received": 30,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 25, "max": 40},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 72,
            "search_appearance_30d": 10,
            "saved_by_recruiters_30d": 3,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.7,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }


# ─── Validation Tests ─────────────────────────────────────────────────────────

class TestValidateCandidate:

    def test_valid_candidate_passes(self):
        c = make_valid_candidate()
        valid, errors = validate_candidate(c)
        assert valid, f"Expected valid, got errors: {errors}"
        assert errors == []

    def test_missing_top_level_key(self):
        c = make_valid_candidate()
        del c["career_history"]
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("career_history" in e for e in errors)

    def test_invalid_candidate_id_format(self):
        c = make_valid_candidate()
        c["candidate_id"] = "CAND_123"  # only 3 digits, not 7
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("candidate_id" in e for e in errors)

    def test_missing_profile_field(self):
        c = make_valid_candidate()
        del c["profile"]["current_title"]
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("current_title" in e for e in errors)

    def test_yoe_out_of_range(self):
        c = make_valid_candidate()
        c["profile"]["years_of_experience"] = 99  # > 50
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("years_of_experience" in e for e in errors)

    def test_invalid_skill_proficiency(self):
        c = make_valid_candidate()
        c["skills"][0]["proficiency"] = "guru"  # not in valid set
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("proficiency" in e for e in errors)

    def test_invalid_recruiter_response_rate(self):
        c = make_valid_candidate()
        c["redrob_signals"]["recruiter_response_rate"] = 1.5  # > 1
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("recruiter_response_rate" in e for e in errors)

    def test_missing_signal_field(self):
        c = make_valid_candidate()
        del c["redrob_signals"]["github_activity_score"]
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("github_activity_score" in e for e in errors)

    def test_invalid_work_mode(self):
        c = make_valid_candidate()
        c["redrob_signals"]["preferred_work_mode"] = "moonwalk"
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("preferred_work_mode" in e for e in errors)

    def test_empty_career_history(self):
        c = make_valid_candidate()
        c["career_history"] = []
        valid, errors = validate_candidate(c)
        assert not valid
        assert any("career_history" in e for e in errors)

    def test_github_minus_one_is_valid(self):
        """github_activity_score = -1 is the 'no GitHub linked' sentinel — must be valid."""
        c = make_valid_candidate()
        c["redrob_signals"]["github_activity_score"] = -1
        valid, errors = validate_candidate(c)
        assert valid, f"Expected valid for github=-1, got: {errors}"


# ─── Streaming Tests ──────────────────────────────────────────────────────────

class TestStreamCandidates:

    def _write_jsonl(self, records, path):
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_streams_valid_records(self, tmp_path):
        records = [make_valid_candidate(f"CAND_{i:07d}") for i in range(1, 4)]
        fpath = tmp_path / "test.jsonl"
        self._write_jsonl(records, fpath)

        result = list(stream_candidates(fpath, validate=True))
        assert len(result) == 3
        assert result[0]["candidate_id"] == "CAND_0000001"

    def test_skips_malformed_json(self, tmp_path):
        fpath = tmp_path / "test.jsonl"
        with open(fpath, "w") as f:
            f.write(json.dumps(make_valid_candidate("CAND_0000001")) + "\n")
            f.write("{BAD JSON\n")
            f.write(json.dumps(make_valid_candidate("CAND_0000003")) + "\n")

        result = list(stream_candidates(fpath, validate=True))
        assert len(result) == 2  # skipped the bad line

    def test_max_records_respected(self, tmp_path):
        records = [make_valid_candidate(f"CAND_{i:07d}") for i in range(1, 11)]
        fpath = tmp_path / "test.jsonl"
        self._write_jsonl(records, fpath)

        result = list(stream_candidates(fpath, validate=True, max_records=5))
        assert len(result) == 5

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            list(stream_candidates(Path("/nonexistent/path.jsonl")))

    def test_invalid_records_skipped_with_validation(self, tmp_path):
        valid   = make_valid_candidate("CAND_0000001")
        invalid = make_valid_candidate("CAND_0000002")
        del invalid["profile"]["current_title"]  # make it invalid

        fpath = tmp_path / "test.jsonl"
        self._write_jsonl([valid, invalid], fpath)

        result = list(stream_candidates(fpath, validate=True))
        assert len(result) == 1
        assert result[0]["candidate_id"] == "CAND_0000001"

    def test_validation_false_passes_invalid_records(self, tmp_path):
        valid   = make_valid_candidate("CAND_0000001")
        invalid = make_valid_candidate("CAND_0000002")
        del invalid["profile"]["current_title"]

        fpath = tmp_path / "test.jsonl"
        self._write_jsonl([valid, invalid], fpath)

        # With validate=False, both records come through
        result = list(stream_candidates(fpath, validate=False))
        assert len(result) == 2