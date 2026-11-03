#!/usr/bin/env bash
# ==============================================================================
# FRAME-AR v2.5: Static UNIX Bash Hypervisor (`frame-orchestrate.sh`)
# ==============================================================================
# Key v2.3 Upgrades:
#  1. Uses `agy --dangerously-skip-permissions -p "<short_prompt>"` inside the
#     isolated `/tmp/frame_jail_*` directory so argv stays <200 bytes.
#  2. Instructs the worker to read `BRIEF.md` and edit `candidate.py` IN-PLACE
#     without running slow shell commands or subagents inside the jail.
#  3. Checks `extract-code` after every attempt (detecting `IN_PLACE_MODIFIED`
#     even if `agy` was interrupted after writing the file).
#  4. Persists all attempt logs and final candidates to
#     `.frame/campaigns/<topic>/logs/step_<N>_<timestamp>/`.
# ==============================================================================

set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOPIC=""
DOMAIN="rl_control"
STEPS=5
AGY_BIN="${AGY_BIN:-agy}"
AGY_FLAGS="${AGY_FLAGS:---dangerously-skip-permissions}"
AGY_TIMEOUT="${AGY_TIMEOUT:-300}"
MAX_DEBUG_RETRIES=3
MAX_STAGNANT_STEPS="${MAX_STAGNANT_STEPS:-5}"
REVIEWER="${USER:-sparsh}"

usage() {
    cat <<EOF
Usage: ./frame-orchestrate.sh --topic <slug> [options]

Options:
  --topic <slug>               Benchmark/campaign topic identifier (required)
  --domain <type>              Domain template if initializing: rl_control | kaggle_tabular (default: rl_control)
  --steps <int>                Number of MCTS swarm steps to execute in Phases 1-4 (default: 5)
  --max-stagnant-steps <int>   Maximum steps without promotion before early-stop (default: 5)
  --agy-timeout <sec>          Hard timeout in seconds per stateless agy invocation (default: 300)
  --reviewer <name>            Human reviewer name for bench.lock (default: \$USER)
  -h, --help                   Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --topic) TOPIC="$2"; shift 2 ;;
        --domain) DOMAIN="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --max-stagnant-steps) MAX_STAGNANT_STEPS="$2"; shift 2 ;;
        --agy-timeout) AGY_TIMEOUT="$2"; shift 2 ;;
        --reviewer) REVIEWER="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "[!] Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$TOPIC" ]]; then
    echo "[!] Error: --topic <slug> is required." >&2
    usage
    exit 1
fi

ACTIVE_JAIL=""
cleanup() {
    if [[ -n "$ACTIVE_JAIL" && -d "$ACTIVE_JAIL" ]]; then
        rm -rf "$ACTIVE_JAIL"
    fi
}
trap cleanup EXIT INT TERM

BENCH_DIR="$WORKSPACE_ROOT/.frame/benchmarks/$TOPIC"
CAMPAIGN_DIR="$WORKSPACE_ROOT/.frame/campaigns/$TOPIC"
SIGNAL_DIR="$CAMPAIGN_DIR/signals"
LOG_ROOT="$CAMPAIGN_DIR/logs"
mkdir -p "$SIGNAL_DIR" "$LOG_ROOT"

# Split AGY_FLAGS into an array safely
read -r -a AGY_FLAG_ARR <<< "$AGY_FLAGS"

echo "======================================================================"
echo " FRAME-AR v2.3 Static Hypervisor"
echo " Workspace : $WORKSPACE_ROOT"
echo " Topic     : $TOPIC ($DOMAIN)"
echo " Target    : $STEPS MCTS Steps | Worker: $AGY_BIN ${AGY_FLAGS} (timeout: ${AGY_TIMEOUT}s)"
echo "======================================================================"

# ==============================================================================
# PHASE 0: BENCHMARK QUALIFICATION, MANUAL WEBAPP AUDIT LOOP, & CRYPTO LOCK
# ==============================================================================
if [[ ! -d "$BENCH_DIR" ]]; then
    echo "[Phase 0] Initializing benchmark directory for '$TOPIC' ($DOMAIN)..."
    python3 -m frame.cli --workspace "$WORKSPACE_ROOT" bench init --topic "$TOPIC" --domain "$DOMAIN"
fi

if python3 -m frame.cli --workspace "$WORKSPACE_ROOT" bench check-lock --topic "$TOPIC" >/dev/null 2>&1; then
    echo "[Phase 0] Valid cryptographic bench.lock verified for '$TOPIC'. Proceeding to Phases 1-4."
else
    echo "[Phase 0] Running initial 6-point empirical meta-verification..."
    python3 -m frame.cli --workspace "$WORKSPACE_ROOT" bench verify --topic "$TOPIC" || true
    python3 -m frame.cli --workspace "$WORKSPACE_ROOT" bench export-stage1 --topic "$TOPIC" >/dev/null 2>&1 || true

    while true; do
        echo ""
        echo "----------------------------------------------------------------------"
        echo " [PHASE 0: MANUAL WEBAPP AUDIT & EDIT MODE — INFINITE WAIT]"
        echo "  1. Webapp Audit Packet : $BENCH_DIR/STAGE1_BLIND_PACKET.md"
        echo "  2. Review Card         : $BENCH_DIR/BENCHMARK_REVIEW.md"
        echo "  3. Editable Files      :"
        echo "       • $BENCH_DIR/bench_spec.json"
        echo "       • $BENCH_DIR/harness.py"
        echo "       • $BENCH_DIR/baseline.py"
        echo "       • $BENCH_DIR/positive_control.py"
        echo "----------------------------------------------------------------------"
        read -r -p "Perform your Webapp audits & file edits, then press [ENTER] to verify (or 'q' to quit): " cmd_in
        if [[ "$cmd_in" == "q" || "$cmd_in" == "Q" ]]; then
            echo "[Phase 0] Exiting hypervisor without locking."
            exit 0
        fi

        echo "[Phase 0] Re-running 6-point empirical verification on your current files..."
        if python3 -m frame.cli --workspace "$WORKSPACE_ROOT" bench verify --topic "$TOPIC"; then
            python3 -m frame.cli --workspace "$WORKSPACE_ROOT" bench export-stage1 --topic "$TOPIC" >/dev/null 2>&1 || true
            echo ""
            cat "$BENCH_DIR/BENCHMARK_REVIEW.md"
            echo ""
            read -r -p "[Human Gate] All 6 checks PASSED. Generate bench.lock and start Swarm Loop? [y/N]: " lock_ans
            if [[ "$lock_ans" =~ ^[Yy]$ ]]; then
                python3 -m frame.cli --workspace "$WORKSPACE_ROOT" bench approve \
                    --topic "$TOPIC" \
                    --reviewer "$REVIEWER" \
                    --notes "Manual webapp cross-audit and empirical verification passed"
                echo "[Phase 0] Benchmark locked successfully."
                break
            else
                echo "[Phase 0] Returning to Manual Edit Mode..."
            fi
        else
            echo ""
            echo "[!] PHASE 0 VERIFICATION FAILED on edited files (syntax, guardrail, competence floor, or variance error)."
            if [[ -f "$BENCH_DIR/BENCHMARK_REVIEW.md" ]]; then
                cat "$BENCH_DIR/BENCHMARK_REVIEW.md"
            fi
            echo "[!] Please fix the reported issues in $BENCH_DIR and press [ENTER] to re-test."
        fi
    done
fi

# ==============================================================================
# PHASES 1–4: JAILED STATELESS SWARM LOOP (Short-Argv + In-Place File Edit)
# ==============================================================================
for (( step=1; step<=STEPS; step++ )); do
    if [[ -f "$SIGNAL_DIR/pause" ]]; then
        echo "[Hypervisor] Pause signal detected ($SIGNAL_DIR/pause)."
        read -r -p "Press [ENTER] to clear pause and continue (or 'q' to stop campaign): " p_ans
        rm -f "$SIGNAL_DIR/pause"
        [[ "$p_ans" == "q" || "$p_ans" == "Q" ]] && break
    fi

    python3 -m frame.cli --workspace "$WORKSPACE_ROOT" bench check-lock --topic "$TOPIC" >/dev/null

    ACTIVE_JAIL="$(mktemp -d "/tmp/frame_jail_${TOPIC}_step${step}_XXXXXX")"
    chmod 700 "$ACTIVE_JAIL"
    STEP_LOG_DIR="$LOG_ROOT/step_${step}_$(date +%H%M%S)"
    mkdir -p "$STEP_LOG_DIR"

    python3 -m frame.cli --workspace "$WORKSPACE_ROOT" prepare-jail \
        --topic "$TOPIC" \
        --jail-dir "$ACTIVE_JAIL"

    # shellcheck disable=SC1090
    source "$ACTIVE_JAIL/meta.env"

    if [[ "${UNCONSUMED_DIRECTIVES:-0}" -eq 0 && "${STAGNANT_STEPS:-0}" -ge "$MAX_STAGNANT_STEPS" ]]; then
        echo ""
        echo "======================================================================"
        echo " [Convergence Early-Stop] $STAGNANT_STEPS consecutive steps completed"
        echo " without a new promoted discovery (max_stagnant_steps=$MAX_STAGNANT_STEPS)."
        echo " Halting MCTS search to prevent deep-tree thrashing and proceeding to Vault Audit."
        echo "======================================================================"
        rm -rf "$ACTIVE_JAIL"
        rmdir "$STEP_LOG_DIR" 2>/dev/null || true
        ACTIVE_JAIL=""
        break
    fi
    cp "$ACTIVE_JAIL/candidate.py" "$ACTIVE_JAIL/parent_backup.py"
    cp "$ACTIVE_JAIL/BRIEF.md" "$STEP_LOG_DIR/BRIEF.md"

    echo ""
    echo "======================================================================"
    echo " [Step $step / $STEPS] Parent: $PARENT_ID | Operator: $ROLE ${DIRECTIVE_ID:+| Directive: $DIRECTIVE_ID}"
    echo " Jail Dir: $ACTIVE_JAIL | Logs: $STEP_LOG_DIR"
    echo "======================================================================"

    # Short <200-byte prompt: avoids argv limits and stops agy from running bash test loops
    PROMPT_MAIN="Read BRIEF.md and edit candidate.py IN-PLACE in the current directory to implement operator '$ROLE'. Do NOT create new files, do NOT run shell/python commands, and exit immediately after saving candidate.py."

    echo "  -> Spawning stateless $AGY_BIN worker..."
    (
        cd "$ACTIVE_JAIL" && \
        timeout --kill-after=5s "${AGY_TIMEOUT}s" \
        "$AGY_BIN" "${AGY_FLAG_ARR[@]}" -p "$PROMPT_MAIN" </dev/null >"$ACTIVE_JAIL/agy_stdout.log" 2>"$ACTIVE_JAIL/agy_stderr.log"
    ) || echo "  [!] Note: $AGY_BIN exited non-zero or hit timeout; checking if candidate.py was modified..."

    cp "$ACTIVE_JAIL/agy_stdout.log" "$STEP_LOG_DIR/attempt1_stdout.log" 2>/dev/null || true
    cp "$ACTIVE_JAIL/agy_stderr.log" "$STEP_LOG_DIR/attempt1_stderr.log" 2>/dev/null || true

    if EXTRACT_MODE=$(python3 -m frame.cli --workspace "$WORKSPACE_ROOT" extract-code \
        --candidate "$ACTIVE_JAIL/candidate.py" \
        --backup "$ACTIVE_JAIL/parent_backup.py" \
        --stdout-log "$ACTIVE_JAIL/agy_stdout.log" 2>&1); then
        echo "  [+] Candidate code captured ($EXTRACT_MODE)."
    else
        echo "  [!] Attempt 1 did not modify candidate.py ($EXTRACT_MODE)."
    fi

    # Gate 0 Fast-Debug Loop (Up to MAX_DEBUG_RETRIES inside the jail)
    GATE0_PASSED=0
    for (( retry=1; retry<=MAX_DEBUG_RETRIES; retry++ )); do
        if python3 -m frame.cli --workspace "$WORKSPACE_ROOT" smoke-check \
            --topic "$TOPIC" \
            --candidate "$ACTIVE_JAIL/candidate.py" >"$ACTIVE_JAIL/SMOKE_ERROR.log" 2>&1; then
            echo "  [✓] Gate 0 Smoke Check Passed (Attempt $retry): $(cat "$ACTIVE_JAIL/SMOKE_ERROR.log")"
            GATE0_PASSED=1
            break
        else
            cp "$ACTIVE_JAIL/SMOKE_ERROR.log" "$STEP_LOG_DIR/attempt${retry}_smoke_error.log" 2>/dev/null || true
            echo "  [X] Gate 0 Failed (Attempt $retry / $MAX_DEBUG_RETRIES):"
            sed 's/^/      /' "$ACTIVE_JAIL/SMOKE_ERROR.log" | tail -n 12
            if [[ $retry -lt $MAX_DEBUG_RETRIES ]]; then
                echo "  -> Spawning stateless $AGY_BIN debug pass..."
                PROMPT_DEBUG="Gate 0 failed. Read SMOKE_ERROR.log and BRIEF.md in the current directory. Edit candidate.py IN-PLACE to fix the error and ensure it is not identical to parent_backup.py. Do NOT run shell commands; exit immediately after saving candidate.py."
                cp "$ACTIVE_JAIL/candidate.py" "$ACTIVE_JAIL/parent_backup.py"
                (
                    cd "$ACTIVE_JAIL" && \
                    timeout --kill-after=5s "${AGY_TIMEOUT}s" \
                    "$AGY_BIN" "${AGY_FLAG_ARR[@]}" -p "$PROMPT_DEBUG" </dev/null >"$ACTIVE_JAIL/agy_stdout.log" 2>"$ACTIVE_JAIL/agy_stderr.log"
                ) || true
                cp "$ACTIVE_JAIL/agy_stdout.log" "$STEP_LOG_DIR/attempt$((retry+1))_stdout.log" 2>/dev/null || true
                cp "$ACTIVE_JAIL/agy_stderr.log" "$STEP_LOG_DIR/attempt$((retry+1))_stderr.log" 2>/dev/null || true
                if RETRY_MODE=$(python3 -m frame.cli --workspace "$WORKSPACE_ROOT" extract-code \
                    --candidate "$ACTIVE_JAIL/candidate.py" \
                    --backup "$ACTIVE_JAIL/parent_backup.py" \
                    --stdout-log "$ACTIVE_JAIL/agy_stdout.log" 2>&1); then
                    echo "  [+] Retry candidate code captured ($RETRY_MODE)."
                else
                    echo "  [!] Retry worker did not modify candidate.py ($RETRY_MODE)."
                fi
            fi
        fi
    done

    if [[ $GATE0_PASSED -eq 0 ]]; then
        echo "  [!] Skipping Gate 1 & Gate 2 for Step $step because Gate 0 failed after $MAX_DEBUG_RETRIES attempts (tree not polluted)."
        rm -rf "$ACTIVE_JAIL"
        ACTIVE_JAIL=""
        continue
    fi

    cp "$ACTIVE_JAIL/candidate.py" "$STEP_LOG_DIR/candidate_final.py" 2>/dev/null || true

    # Gate 1 (Dev-Diagnostic) & Gate 2 (Confirmatory Fresh Seeds + LORD-3)
    echo "  -> Executing Trusted Host Funnel (Gate 1 & Gate 2)..."
    STEP_ARGS=(
        --workspace "$WORKSPACE_ROOT"
        step
        --topic "$TOPIC"
        --candidate "$ACTIVE_JAIL/candidate.py"
        --parent "$PARENT_ID"
        --role "$ROLE"
    )
    if [[ -n "$DIRECTIVE_ID" ]]; then
        STEP_ARGS+=(--directive "$DIRECTIVE_ID")
    fi

    python3 -m frame.cli "${STEP_ARGS[@]}" || true

    rm -rf "$ACTIVE_JAIL"
    ACTIVE_JAIL=""
done

# ==============================================================================
# PHASE 5: FINAL UNTOUCHED VAULT-HOLDOUT AUDIT
# ==============================================================================
echo ""
echo "======================================================================"
echo " [Phase 5] Running Final Untouched Vault-Holdout Audit..."
echo "======================================================================"
python3 -m frame.cli --workspace "$WORKSPACE_ROOT" vault-eval --topic "$TOPIC" || true
python3 -m frame.cli --workspace "$WORKSPACE_ROOT" status --topic "$TOPIC"
