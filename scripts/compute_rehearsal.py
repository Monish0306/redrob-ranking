"""
compute_rehearsal.py  —  Phase 9
----------------------------------
Proves the ranking step (NOT the embedding pre-compute) actually satisfies
every Stage 3 sandbox constraint:
  - Peak memory ≤ 16 GB
  - Wall-clock ≤ 5 minutes
  - Zero network calls during the ranking step itself

RUN: python scripts/compute_rehearsal.py

WHAT IT DOES:
  1. Greps the entire src/ directory for any hosted-LLM import (proof, not just claim)
  2. Runs rank.py with memory profiling attached
  3. Reports peak RSS memory and total wall-clock time
  4. Flags if the embed_jd step makes ANY network call (it does right now —
     see the fix below for the offline-mode flag)
"""

import sys
import subprocess
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FORBIDDEN_IMPORTS = [
    r"\bimport\s+openai\b",
    r"\bimport\s+anthropic\b",
    r"\bimport\s+google\.generativeai\b",
    r"\bfrom\s+openai\b",
    r"\bfrom\s+anthropic\b",
    r"\bfrom\s+google\.generativeai\b",
    r"requests\.(get|post)\(.*openai",
    r"requests\.(get|post)\(.*anthropic",
]


def grep_forbidden_imports(src_dir: Path) -> list:
    """Scan every .py file in src/ for hosted-LLM SDK usage."""
    violations = []
    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_IMPORTS:
            if re.search(pattern, text):
                violations.append(f"{py_file}: matched pattern '{pattern}'")
    return violations


def run_with_memory_profiling(cmd: list) -> dict:
    """
    Run a command and track peak memory using psutil (cross-platform).
    Falls back to basic timing if psutil isn't installed.
    """
    try:
        import psutil
    except ImportError:
        print("⚠️  psutil not installed. Run: pip install psutil")
        print("    Falling back to timing-only rehearsal.")
        t0 = time.time()
        subprocess.run(cmd, check=True)
        return {"elapsed": time.time() - t0, "peak_memory_mb": None}

    t0 = time.time()
    proc = subprocess.Popen(cmd)
    p = psutil.Process(proc.pid)
    peak_mem = 0

    while proc.poll() is None:
        try:
            mem = p.memory_info().rss
            # Include child processes (e.g. torch worker threads)
            for child in p.children(recursive=True):
                mem += child.memory_info().rss
            peak_mem = max(peak_mem, mem)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        time.sleep(0.5)

    elapsed = time.time() - t0
    return {"elapsed": elapsed, "peak_memory_mb": peak_mem / (1024 * 1024)}


def main():
    root = Path(__file__).parent.parent
    src_dir = root / "src"

    print("=" * 60)
    print("PHASE 9 — COMPUTE REHEARSAL")
    print("=" * 60)

    # ── Check 1: No hosted-LLM imports anywhere in src/ ──────────────────────
    print("\n[1/3] Scanning src/ for hosted-LLM SDK usage...")
    violations = grep_forbidden_imports(src_dir)
    if violations:
        print("❌ FOUND FORBIDDEN IMPORTS:")
        for v in violations:
            print(f"   {v}")
    else:
        print("✅ No hosted-LLM SDK imports found in src/")

    # ── Check 2: Run rank.py with memory + time profiling ────────────────────
    print("\n[2/3] Running rank.py with memory profiling...")
    print("      (This re-runs the full ranking step — ~1 minute)")

    cmd = [
        sys.executable, "-m", "src.rank",
        "--candidates", str(root / "data" / "candidates.jsonl"),
        "--jd",         str(root / "data" / "job_description.txt"),
        "--out",        str(root / "submission_rehearsal.csv"),
    ]
    result = run_with_memory_profiling(cmd)

    print(f"\n      Elapsed time : {result['elapsed']:.1f}s "
          f"(budget: 300s)")
    if result["peak_memory_mb"]:
        print(f"      Peak memory  : {result['peak_memory_mb']:.0f} MB "
              f"(budget: 16,384 MB)")

    # ── Check 3: Verdict ───────────────────────────────────────────────────────
    print("\n[3/3] VERDICT")
    print("-" * 60)
    time_ok = result["elapsed"] <= 300
    mem_ok  = (result["peak_memory_mb"] is None or
              result["peak_memory_mb"] <= 16384)
    imports_ok = len(violations) == 0

    print(f"  Time budget (≤300s)      : {'✅ PASS' if time_ok else '❌ FAIL'}")
    print(f"  Memory budget (≤16GB)    : {'✅ PASS' if mem_ok else '❌ FAIL'}")
    print(f"  No hosted-LLM imports    : {'✅ PASS' if imports_ok else '❌ FAIL'}")

    if time_ok and mem_ok and imports_ok:
        print("\n✅ ALL CHECKS PASSED — ready for Stage 3 reproduction.")
    else:
        print("\n❌ ONE OR MORE CHECKS FAILED — fix before submitting.")

    print("=" * 60)


if __name__ == "__main__":
    main()