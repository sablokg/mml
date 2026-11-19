#!/bin/bash
# Real-data benchmark: Zhu et al. 2020 (schizophrenia vs healthy controls), genus and species level.
# usage: ./run_all.sh [reps=3] [auto=1]      (about 15-20 min single-threaded with AUTO)
set -e
cd "$(dirname "$0")"; mkdir -p results
REPS=${1:-3}; AUTO=${2:-1}
for L in genus species; do
  cargo run --release --features xgboost --example zhu_bench -- data/$L.tsv results/res_$L.csv results/folds_$L.csv $REPS $AUTO 2> results/log_$L.txt
  python3 sk_ref.py $L
done
python3 analyze.py
