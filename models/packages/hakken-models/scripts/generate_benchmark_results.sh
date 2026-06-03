#!/usr/bin/env bash

for year in 1990 2000 2010 2020; do
    echo "Running ${year}"
    extra=()
    [[ "${year}" != "1990" ]] && extra=("bar.show_title=false")
    uv run python scripts/generate_benchmark_results.py \
        "paths.results_csv=results/benchmark_v2/${year}/comparison_summary.csv" \
        "paths.output_dir=results/benchmark_v2" \
        "paths.label=${year}" \
        "radar.title=${year}" \
        "bar.shared_ylim=true" \
        "bar.ylim=[0,100]" \
        "${extra[@]}"
    echo
done