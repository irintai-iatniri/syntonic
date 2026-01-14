#!/bin/bash
# Overnight validation experiments for syntonic softmax

cd "/home/Andrew/Documents/SRT Complete/implementation/syntonic"

RESULTS_DIR="python/syntonic/experiments/results"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "SYNTONIC SOFTMAX OVERNIGHT VALIDATION"
echo "Started: $(date)"
echo "========================================"

# Experiment 1A: XOR (50 generations)
echo ""
echo "Running Experiment 1A: XOR Winding Classification..."
python -u -m syntonic.experiments.validate_syntonic_softmax \
    --experiment xor \
    --max-generations 50 \
    --base-dim 32 \
    --num-blocks 3 \
    2>&1 | tee "$RESULTS_DIR/xor_overnight.log"

# Experiment 1B: Particles (30 generations, leave-one-out)
echo ""
echo "Running Experiment 1B: Particle Classification..."
python -u -m syntonic.experiments.validate_syntonic_softmax \
    --experiment particles \
    --max-generations 30 \
    --base-dim 32 \
    --num-blocks 3 \
    2>&1 | tee "$RESULTS_DIR/particles_overnight.log"

# Experiment 1C: CSV Dataset (50 generations)
echo ""
echo "Running Experiment 1C: CSV Dataset Classification..."
python -u -m syntonic.experiments.validate_syntonic_softmax \
    --experiment csv \
    --max-generations 50 \
    --base-dim 32 \
    --num-blocks 3 \
    2>&1 | tee "$RESULTS_DIR/csv_overnight.log"

echo ""
echo "========================================"
echo "OVERNIGHT VALIDATION COMPLETE"
echo "Finished: $(date)"
echo "========================================"
