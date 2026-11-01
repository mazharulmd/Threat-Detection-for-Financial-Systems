#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. HALX_THREADS=${HALX_THREADS:-2}
python3 -m experiments.run ablation --dataset unsw --seeds 0
python3 -m experiments.run privacy  --dataset cic  --seeds 0 --model HALX
