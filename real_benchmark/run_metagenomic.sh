#!/bin/bash
# Benchmark all public metagenomic datasets (data/*.tsv built by prep_metagenomic.py), same pipelines/folds protocol as run_all.sh.
# usage: ./run_metagenomic.sh [reps=3] [auto=1] [dataset ...]      (default: all tables listed in data/datasets_summary.csv)
set -e
cd "$(dirname "$0")"; mkdir -p results
REPS=${1:-3}; AUTO=${2:-1}; shift 2 || true
SETS=${@:-$(tail -n +2 data/datasets_summary.csv | cut -d, -f1)}
cargo build --release --features xgboost --example zhu_bench --manifest-path ../Cargo.toml 2>&1 | tail -1
for L in $SETS; do
  [ -f results/res_$L.csv ] && [ -f results/sk_$L.csv ] && { echo "skip $L (done)"; continue; }
  ../target/release/examples/zhu_bench data/$L.tsv results/res_$L.csv results/folds_$L.csv $REPS $AUTO 2> results/log_$L.txt
  python3 sk_ref.py $L
done
python3 analyze.py
