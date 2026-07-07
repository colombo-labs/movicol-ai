#!/bin/bash
# MoviCol — Load & Stress Test Runner
# Prereqs: locust installed, AI service running on localhost:8000
#
# Usage:
#   ./tests/performance/run_tests.sh load     # Normal load test (50 users, 60s)
#   ./tests/performance/run_tests.sh stress   # Stress test (200 users, 120s)
#   ./tests/performance/run_tests.sh both     # Run both sequentially

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
HOST="${HOST:-http://localhost:8000}"
LOCUSTFILE="$SCRIPT_DIR/locustfile.py"

mkdir -p "$RESULTS_DIR"

echo "🧪 MoviCol Performance Tests"
echo "   Host: $HOST"
echo "   Results: $RESULTS_DIR"
echo ""

# Verify service is up
if ! curl -s "$HOST/health" > /dev/null 2>&1; then
    echo "❌ AI service not running at $HOST"
    echo "   Start it with: cd movicol-ai && uvicorn app.main:app --port 8000"
    exit 1
fi
echo "✅ AI service is up"
echo ""

run_load() {
    echo "═══════════════════════════════════════"
    echo "📊 LOAD TEST — 50 users, spawn 5/s, 60s"
    echo "═══════════════════════════════════════"
    locust -f "$LOCUSTFILE" \
        --host "$HOST" \
        --users 50 \
        --spawn-rate 5 \
        --run-time 60s \
        --headless \
        --csv "$RESULTS_DIR/load" \
        --only-summary \
        --loglevel WARNING \
        -L ChatUser
    echo ""
    echo "📈 Results saved to $RESULTS_DIR/load_stats.csv"
    echo ""
}

run_stress() {
    echo "═══════════════════════════════════════"
    echo "🔥 STRESS TEST — 200 users, spawn 20/s, 120s"
    echo "═══════════════════════════════════════"
    locust -f "$LOCUSTFILE" \
        --host "$HOST" \
        --users 200 \
        --spawn-rate 20 \
        --run-time 120s \
        --headless \
        --csv "$RESULTS_DIR/stress" \
        --only-summary \
        --loglevel WARNING \
        -L StressUser
    echo ""
    echo "📈 Results saved to $RESULTS_DIR/stress_stats.csv"
    echo ""
}

case "${1:-both}" in
    load)
        run_load
        ;;
    stress)
        run_stress
        ;;
    both)
        run_load
        echo "Cooling down 5s..."
        sleep 5
        run_stress
        ;;
    *)
        echo "Usage: $0 {load|stress|both}"
        exit 1
        ;;
esac

echo "═══════════════════════════════════════"
echo "✅ Performance tests complete!"
echo ""
echo "View results:"
echo "  cat $RESULTS_DIR/load_stats.csv"
echo "  cat $RESULTS_DIR/stress_stats.csv"
echo ""
echo "Or run with UI:"
echo "  locust -f $LOCUSTFILE --host $HOST"
echo "  → Open http://localhost:8089"
