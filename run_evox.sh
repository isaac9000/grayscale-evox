#!/usr/bin/env bash
# Run SkyDiscover/EvoX for 25 iterations, but first verify the H100 baseline
# is within TARGET_LOW–TARGET_HIGH µs (proxy for consistent node).
# Retries up to MAX_ATTEMPTS times before giving up.

set -uo pipefail

TARGET_LOW=95.0
TARGET_HIGH=108.0
MAX_ATTEMPTS=10
RUN_BASE="grayscale/skydiscover_runs"
PYTHON=".venv/bin/python"

set -a
source .env
set +a

mkdir -p "$RUN_BASE"

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    echo ""
    echo "=== Attempt $attempt / $MAX_ATTEMPTS : checking baseline ==="

    BASELINE_JSON=$(mktemp /tmp/baseline_XXXXXX.json)

    if ! $PYTHON grayscale/run_eval.py grayscale/starting_point.py \
            -o "$BASELINE_JSON" --mode leaderboard; then
        echo "  Baseline eval failed — retrying in 20s..."
        rm -f "$BASELINE_JSON"
        sleep 20
        continue
    fi

    PARSE=$($PYTHON -c "
import json, re
md = json.load(open('$BASELINE_JSON'))
gm = re.search(r'Geometric mean: ⏱ ([\d.]+)', md)
gpu = re.search(r'GPU: \`([^\`]+)\`', md)
print(gm.group(1) if gm else '0')
print(gpu.group(1) if gpu else 'unknown')
")
    rm -f "$BASELINE_JSON"

    GEOMEAN=$(echo "$PARSE" | sed -n '1p')
    GPU_NAME=$(echo "$PARSE" | sed -n '2p')

    echo "  GPU         : $GPU_NAME"
    echo "  Geomean     : ${GEOMEAN} µs   (target: ${TARGET_LOW}–${TARGET_HIGH} µs)"

    IN_RANGE=$($PYTHON -c "
g = float('$GEOMEAN')
print('yes' if $TARGET_LOW <= g <= $TARGET_HIGH else 'no')
")

    if [ "$IN_RANGE" = "yes" ]; then
        echo "  ✅ Baseline accepted — starting 25-iteration EvoX run"

        RUN_NUM=$(ls -d "$RUN_BASE"/run* 2>/dev/null | wc -l)
        RUN_NUM=$((RUN_NUM + 1))
        RUN_OUT="$RUN_BASE/run$RUN_NUM"
        mkdir -p "$RUN_OUT"

        tmux kill-session -t evox-grayscale 2>/dev/null || true
        tmux new-session -d -s evox-grayscale

        tmux send-keys -t evox-grayscale \
            "cd /workspace/grayscale-openevolve && set -a && source .env && set +a && \
.venv/bin/skydiscover-run \
  grayscale/starting_point.py \
  grayscale/skydiscover_evaluator.py \
  --config grayscale/skydiscover_config.yaml \
  --search evox \
  --iterations 25 \
  --output $RUN_OUT \
2>&1 | tee ${RUN_OUT}.log" Enter

        echo ""
        echo "  tmux session : evox-grayscale"
        echo "  Output dir   : $RUN_OUT"
        echo "  Log          : ${RUN_OUT}.log"
        echo ""
        echo "  Monitor : tmux attach -t evox-grayscale"
        exit 0
    else
        if [ "$attempt" -lt "$MAX_ATTEMPTS" ]; then
            echo "  ❌ Out of range — waiting 30s..."
            sleep 30
        else
            echo "  ❌ Gave up after $MAX_ATTEMPTS attempts. Last: ${GEOMEAN} µs on $GPU_NAME"
            exit 1
        fi
    fi
done
