"""Benchmark registry, 3-stage cross-family audit packet generator, and cryptographic lock."""

import json
from pathlib import Path
from typing import Literal, Optional

from frame.bench.templates import get_kaggle_tabular_template, get_rl_control_template
from frame.bench.verifier import verify_benchmark
from frame.schemas import (
    BenchLock,
    BenchSpec,
    BenchVerificationReport,
    CommitteeAuditRecord,
    TrustGrade,
    utc_now_iso,
)
from frame.utils.canonical_json import compute_file_sha256


def get_bench_dir(workspace_root: Path, topic: str) -> Path:
    slug = topic.strip().lower().replace(" ", "_")
    return workspace_root.resolve() / ".frame" / "benchmarks" / slug


get_benchmark_dir = get_bench_dir


def init_benchmark(
    workspace_root: Path,
    topic: str,
    domain: Literal["kaggle_tabular", "rl_control"],
) -> Path:
    """Initializes a Phase 0 benchmark directory from domain templates."""
    slug = topic.strip().lower().replace(" ", "_")
    bench_dir = get_bench_dir(workspace_root, slug)
    bench_dir.mkdir(parents=True, exist_ok=True)

    files = (
        get_kaggle_tabular_template(slug)
        if domain == "kaggle_tabular"
        else get_rl_control_template(slug)
    )
    for filename, content in files.items():
        (bench_dir / filename).write_text(content, encoding="utf-8")

    committee_path = bench_dir / "committee_audit.json"
    if not committee_path.exists():
        committee_path.write_text(
            CommitteeAuditRecord().model_dump_json(indent=2), encoding="utf-8"
        )
    return bench_dir


def _load_committee(bench_dir: Path) -> CommitteeAuditRecord:
    p = bench_dir / "committee_audit.json"
    if not p.exists():
        return CommitteeAuditRecord()
    return CommitteeAuditRecord.model_validate_json(p.read_text(encoding="utf-8"))


def _save_committee(bench_dir: Path, rec: CommitteeAuditRecord) -> None:
    (bench_dir / "committee_audit.json").write_text(
        rec.model_dump_json(indent=2), encoding="utf-8"
    )


def export_stage1_blind_packet(bench_dir: Path) -> Path:
    """Generates the Stage 1 Blind Audit Packet (code + empirical numbers) for Gemini Frontier & Claude."""
    rep_path = bench_dir / "verification_report.json"
    if not rep_path.exists():
        verify_benchmark(bench_dir)

    spec_txt = (bench_dir / "bench_spec.json").read_text(encoding="utf-8")
    rep_txt = rep_path.read_text(encoding="utf-8")
    harness_txt = (bench_dir / "harness.py").read_text(encoding="utf-8")
    base_txt = (bench_dir / "baseline.py").read_text(encoding="utf-8")
    pos_txt = (bench_dir / "positive_control.py").read_text(encoding="utf-8")

    packet = f"""# STAGE 1 BLIND BENCHMARK AUDIT PACKET
Perform an independent, blind methodological critique of this Phase 0 benchmark.
Evaluate:
1. Construct validity: Does `harness.py` measure genuine domain progress without data/reward leakage?
2. Baseline competence: Is `baseline.py` a fair reference anchor rather than a weak strawman?
3. Guardrail sufficiency (strictly capped at <= 4 guardrails): Are any critical failure modes unguarded?
4. Statistical signal-to-noise: Inspect `empirical_sigma_d` vs `rope_epsilon` in the empirical report below.

## 1. Empirical `frame bench verify` Report
```json
{rep_txt}
```

## 2. `bench_spec.json`
```json
{spec_txt}
```

## 3. `harness.py`
```python
{harness_txt}
```

## 4. `baseline.py` & `positive_control.py`
```python
# --- baseline.py ---
{base_txt}

# --- positive_control.py ---
{pos_txt}
```
"""
    out_path = bench_dir / "STAGE1_BLIND_PACKET.md"
    out_path.write_text(packet, encoding="utf-8")
    return out_path


def record_stage1_review(
    bench_dir: Path,
    model_family: Literal["gemini", "claude"],
    review_text: str,
) -> CommitteeAuditRecord:
    rec = _load_committee(bench_dir)
    if model_family == "gemini":
        rec.stage1_gemini_blind_review = review_text.strip()
    else:
        rec.stage1_claude_blind_review = review_text.strip()
    _save_committee(bench_dir, rec)
    return rec


def export_stage2_cross_packet(
    bench_dir: Path,
    target_model: Literal["gemini", "claude"],
) -> Path:
    """Generates the Stage 2 Cross-Critique Packet swapping Gemini and Claude blind evaluations."""
    rec = _load_committee(bench_dir)
    if not rec.stage1_gemini_blind_review or not rec.stage1_claude_blind_review:
        raise RuntimeError(
            "Both Gemini and Claude Stage 1 blind reviews must be recorded before exporting Stage 2 cross-critique packets."
        )

    own_review = rec.stage1_gemini_blind_review if target_model == "gemini" else rec.stage1_claude_blind_review
    peer_review = rec.stage1_claude_blind_review if target_model == "gemini" else rec.stage1_gemini_blind_review
    peer_name = "Claude" if target_model == "gemini" else "Gemini"

    packet = f"""# STAGE 2 CROSS-MODEL CRITIQUE PACKET (FOR {target_model.upper()})
Below is your Stage 1 blind evaluation alongside the independent Stage 1 evaluation from {peer_name}.
1. Identify blind spots or valid failure modes caught by {peer_name} that you missed.
2. Flag any incorrect claims or unnecessary guardrail bloat proposed by {peer_name} (remember: max 4 guardrails total).
3. Emit a concise list of high-priority adjustments for the human reviewer.

## Your Stage 1 Blind Review ({target_model.upper()})
{own_review}

## Peer Stage 1 Blind Review ({peer_name.upper()})
{peer_review}
"""
    out_path = bench_dir / f"STAGE2_CROSS_FOR_{target_model.upper()}.md"
    out_path.write_text(packet, encoding="utf-8")
    return out_path


def record_stage2_critique(
    bench_dir: Path,
    model_family: Literal["gemini", "claude"],
    critique_text: str,
) -> CommitteeAuditRecord:
    rec = _load_committee(bench_dir)
    if model_family == "gemini":
        rec.stage2_gemini_on_claude_critique = critique_text.strip()
    else:
        rec.stage2_claude_on_gemini_critique = critique_text.strip()
    _save_committee(bench_dir, rec)
    return rec


def generate_human_review_markdown(bench_dir: Path) -> Path:
    """Emits the one-page `BENCHMARK_REVIEW.md` human gate package."""
    spec = BenchSpec.model_validate_json((bench_dir / "bench_spec.json").read_text(encoding="utf-8"))
    rep = BenchVerificationReport.model_validate_json(
        (bench_dir / "verification_report.json").read_text(encoding="utf-8")
    )
    rec = _load_committee(bench_dir)

    guardrail_rows = "\n".join(
        f"| `{g.name}` | `{g.op} {g.threshold}` | {g.description} |" for g in spec.guardrails
    )

    md = f"""# BENCHMARK REVIEW PACKAGE: `{spec.benchmark_id}`
- **Trust Grade**: **{rep.trust_grade.value}**
- **Domain**: `{spec.domain}` | **Primary Metric**: `{spec.primary_metric_name}` (`{spec.direction.value}`)
- **Estimator**: `{spec.estimator.value}` | **ROPE**: `{spec.rope_epsilon}` | **Blend ROPE**: `{spec.blend_rope_epsilon}`
- **External Anchor**: {spec.external_anchor.source_name} ({spec.external_anchor.reference_url_or_citation})

## 1. Meta-Verification Summary
| Check | Passed | Empirical Value |
| :--- | :--- | :--- |
| Smoke Execution | `{rep.smoke_passed}` | Completed within timeout |
| Baseline Competence | `{rep.baseline_competent}` | Mean=`{rep.baseline_mean}`, IQM=`{rep.baseline_iqm}` (Floor=`{spec.external_anchor.competence_floor}`) |
| Null Probe Separation | `{rep.null_separation_passed}` | `{json.dumps(rep.null_probe_scores)}` |
| Positive Control | `{rep.positive_control_passed}` | Mean=`{rep.positive_control_mean}`, IQM=`{rep.positive_control_iqm}` |
| Degradation Ladder | `{rep.sensitivity_monotonic_passed}` | `{json.dumps(rep.degradation_ladder_scores)}` |
| Expressivity Probe | `{rep.expressivity_probe_passed}` | 3 polymorphic policy architectures evaluated |
| Seed Noise & Power | `k* = {rep.recommended_k_star}` | `sigma_D = {rep.empirical_sigma_d}` |

## 2. Active Guardrails ({len(spec.guardrails)} / 4 Max)
| Guardrail | Constraint | Purpose |
| :--- | :--- | :--- |
{guardrail_rows}

## 3. Cross-Family Committee Audit
- **Stage 1 Gemini Blind Review**: {rec.stage1_gemini_blind_review or "Pending"}
- **Stage 1 Claude Blind Review**: {rec.stage1_claude_blind_review or "Pending"}
- **Stage 2 Gemini Cross-Critique**: {rec.stage2_gemini_on_claude_critique or "Pending"}
- **Stage 2 Claude Cross-Critique**: {rec.stage2_claude_on_gemini_critique or "Pending"}
"""
    out = bench_dir / "BENCHMARK_REVIEW.md"
    out.write_text(md, encoding="utf-8")
    return out


def approve_benchmark(
    bench_dir: Path,
    reviewer: str,
    notes: str = "",
    require_committee: bool = False,
) -> BenchLock:
    """Human Gate: Approves the benchmark and writes the cryptographic `bench.lock`."""
    bench_dir = bench_dir.resolve()
    rep_path = bench_dir / "verification_report.json"
    if not rep_path.exists():
        rep = verify_benchmark(bench_dir)
    else:
        rep = BenchVerificationReport.model_validate_json(rep_path.read_text(encoding="utf-8"))

    if rep.trust_grade == TrustGrade.FAIL:
        raise RuntimeError(
            f"Cannot approve benchmark with TrustGrade=FAIL. Reasons: {rep.failure_reasons}"
        )

    rec = _load_committee(bench_dir)
    if require_committee and (
        not rec.stage1_gemini_blind_review
        or not rec.stage1_claude_blind_review
        or not rec.stage2_gemini_on_claude_critique
        or not rec.stage2_claude_on_gemini_critique
    ):
        raise RuntimeError("Committee audit incomplete: Stage 1 and Stage 2 reviews are required.")

    rec.stage3_human_reviewer = reviewer
    rec.stage3_human_notes = notes
    rec.approved_at = utc_now_iso()
    _save_committee(bench_dir, rec)
    generate_human_review_markdown(bench_dir)

    lock = BenchLock(
        benchmark_id=rep.benchmark_id,
        bench_spec_sha256=compute_file_sha256(bench_dir / "bench_spec.json"),
        harness_sha256=compute_file_sha256(bench_dir / "harness.py"),
        baseline_sha256=compute_file_sha256(bench_dir / "baseline.py"),
        positive_control_sha256=compute_file_sha256(bench_dir / "positive_control.py"),
        verification_report_sha256=compute_file_sha256(rep_path),
        trust_grade=rep.trust_grade,
        approved_by=reviewer,
    )
    (bench_dir / "bench.lock").write_text(lock.model_dump_json(indent=2), encoding="utf-8")
    return lock


def verify_bench_lock(bench_dir: Path) -> BenchLock:
    """Verifies that the human-approved `bench.lock` exists and matches all benchmark file hashes."""
    lock_path = bench_dir / "bench.lock"
    if not lock_path.exists():
        raise PermissionError(
            f"Human Gate Locked: No `bench.lock` found in {bench_dir}. "
            "Run `frame bench verify` and `frame bench approve` first."
        )
    lock = BenchLock.model_validate_json(lock_path.read_text(encoding="utf-8"))
    checks = [
        ("bench_spec.json", lock.bench_spec_sha256),
        ("harness.py", lock.harness_sha256),
        ("baseline.py", lock.baseline_sha256),
        ("positive_control.py", lock.positive_control_sha256),
    ]
    for fname, expected_hash in checks:
        actual = compute_file_sha256(bench_dir / fname)
        if actual != expected_hash:
            raise PermissionError(
                f"Benchmark Integrity Violation: `{fname}` hash changed after human lock! "
                f"Expected {expected_hash[:12]}, got {actual[:12]}."
            )
    return lock
