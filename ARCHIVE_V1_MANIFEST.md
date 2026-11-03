# FRAME-AR v1 Preserved State Manifest

> [!CAUTION]
> **OUT OF SUPPORT — RESTORATION STRICTLY NOT RECOMMENDED**
> This manifest preserves the legacy in-process v1 codebase for historical archival purposes only. **Restoring this archive into an active workspace is strictly not recommended due to severe result conflicts, in-process benchmark contamination, and telemetry collisions.** Use FRAME-AR v2.5+ for all empirical research and active campaigns.

- **Archive Date**: 2026-10-29
- **Preserved Git Commit**: `1a55a63` (feat(core): initial release of FRAME-AR autonomous research platform)
- **Git Tag**: `v1.0.0-archive`
- **Git Preservation Branch**: `archive/v1-main`
- **Verification Status**: PASSED (`793 passed in 65.14s`, 100% test pass rate)
- **Archive File**: `frame_v1_preserved_state.tar.xz`
- **Compression**: Tar + XZ (`-9e`, ultra high compression)

## Contents Preserved
- `frame/`: Complete FRAME-AR v1 core python package (agents, brain, feedback, pot, runners, schemas, utils, verification)
- `tests/`: Complete test suite (23 test files, 793 unit and integration tests)
- `docs/`: Technical reports and documentation (LaTeX source and PDF: `FRAME_AR_SYSTEM_REPORT.pdf`)
- `skills/`: Agent skills (`frame-ar`)
- `.agents/`: Agent teamwork records and specs from development track
- `.frame/`: Runtime state and ledgers (`accounting_ledger.json`, `lord_state.json`)
- `ver_c/`: Verification reports and reference versions
- `working/`: Working source and development scratch
- `FRAME_AR_SPECIFICATION.md`: Complete v1 architectural specification
- `ORIGINAL_REQUEST.md`, `PROJECT.md`, `TEST_INFRA.md`, `TEST_READY.md`, `todo.md`
- `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`
- `.git/`: Complete Git version control history and objects

## Restoring the Archive
To restore this exact state into any directory:
```bash
tar -xvf frame_v1_preserved_state.tar.xz -C /path/to/destination
```
To verify test pass post-restore:
```bash
pytest -q
```
