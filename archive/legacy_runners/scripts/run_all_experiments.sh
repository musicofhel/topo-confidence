#!/bin/bash
# Run all experiments sequentially. Survives session drops via nohup.
# Usage: nohup bash scripts/run_all_experiments.sh > experiments.log 2>&1 &

set -e
cd ~/topo-confidence
source .venv/bin/activate

echo "=== EXPERIMENT 1: MATH-500 ==="
echo "Started: $(date)"
python scripts/experiment1_math500.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda
echo "Finished Exp 1: $(date)"
echo ""

echo "=== EXPERIMENT 4: BASELINES ==="
echo "Started: $(date)"
python scripts/experiment4_baselines.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda
echo "Finished Exp 4: $(date)"
echo ""

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

echo "=== EXPERIMENT 3: CROSS-BENCHMARK ==="
echo "Started: $(date)"
python scripts/experiment3_cross_benchmark.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --max-problems 200
echo "Finished Exp 3: $(date)"
echo ""

echo "=== EXPERIMENT 2: CROSS-MODEL ==="
echo "Started: $(date)"
python scripts/experiment2_cross_model.py --device cuda --max-problems 200
echo "Finished Exp 2: $(date)"
echo ""

echo "=== ALL EXPERIMENTS DONE ==="
echo "Finished: $(date)"
