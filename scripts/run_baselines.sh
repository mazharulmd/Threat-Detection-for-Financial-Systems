#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=. HALX_THREADS=${HALX_THREADS:-2}
for ds in cic unsw; do
  python3 -m experiments.run main --dataset $ds --seeds 0 1 2 \
     --methods XGBoost RF IsolationForest MLP CNN-LSTM LSTM Transformer
done
