#!/usr/bin/env bash
# HALX-specific lane (used alongside run_baselines.sh to use all cores).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. HALX_THREADS=${HALX_THREADS:-2}
R="python3 -m experiments.run"
$R main --dataset cic  --seeds 0 1 2 --methods HALX
$R main --dataset unsw --seeds 0 1 2 --methods HALX
$R privacy --dataset cic  --seeds 0 --model XGBoost
$R privacy --dataset unsw --seeds 0 --model XGBoost
$R latency --dataset cic
$R latency --dataset unsw
$R ablation --dataset cic  --seeds 0 1 2
$R privacy --dataset cic  --seeds 0 --model HALX
$R ablation --dataset unsw --seeds 0
