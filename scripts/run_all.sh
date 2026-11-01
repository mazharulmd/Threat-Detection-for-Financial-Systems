#!/usr/bin/env bash
# Runs every experiment reported in the paper (resumable: finished runs are skipped).
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. HALX_THREADS=${HALX_THREADS:-2}
R="python3 -m experiments.run"
# 1) main comparison
$R main --dataset cic  --seeds 0 1 2
$R main --dataset unsw --seeds 0 1 2
# 2) privacy-utility sweeps (XGBoost on both datasets, HALX on CICIDS2017)
$R privacy --dataset cic  --seeds 0 --model XGBoost
$R privacy --dataset unsw --seeds 0 --model XGBoost
$R privacy --dataset cic  --seeds 0 --model HALX
# 3) ablations
$R ablation --dataset cic  --seeds 0 1 2
$R ablation --dataset unsw --seeds 0
# 4) CPU latency / throughput
$R latency --dataset cic
$R latency --dataset unsw
