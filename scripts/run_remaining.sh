#!/bin/bash
# Run remaining experiments. Exp 1 already done.
set -e
cd ~/topo-confidence
source .venv/bin/activate

echo "=== EXPERIMENT 5: SELECTIVE PREDICTION ==="
echo "Started: $(date)"
python scripts/experiment5_selective.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda
echo "Finished Exp 5: $(date)"
echo ""

echo "=== EXPERIMENT 6: SPEED BENCHMARK ==="
echo "Started: $(date)"
python scripts/experiment6_speed.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --n-problems 50
echo "Finished Exp 6: $(date)"
echo ""

echo "=== EXPERIMENT 3: CROSS-BENCHMARK (200 problems each) ==="
echo "Started: $(date)"
python scripts/experiment3_cross_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --max-problems 200
echo "Finished Exp 3: $(date)"
echo ""

echo "=== EXPERIMENT 4: BASELINES (200 problems, logits collected) ==="
echo "Started: $(date)"
python scripts/experiment4_baselines.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --max-problems 200
echo "Finished Exp 4: $(date)"
echo ""

echo "=== EXPERIMENT 2: CROSS-MODEL (200 problems) ==="
echo "Started: $(date)"
python scripts/experiment2_cross_model.py --device cuda --max-problems 200
echo "Finished Exp 2: $(date)"
echo ""

echo "=== ALL REMAINING EXPERIMENTS DONE ==="
echo "Finished: $(date)"
