"""Aggregate experiment results into a summary CSV."""

import argparse

from voc_bench.evaluation.analyze import aggregate_results, summary_table


def main():
    parser = argparse.ArgumentParser(description="Aggregate experiment results")
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Root results directory",
    )
    parser.add_argument(
        "--output", type=str, default="results/summary.csv",
        help="Path to save summary CSV",
    )
    args = parser.parse_args()

    print(f"Scanning {args.results_dir} for eval_results.json files...")
    df = aggregate_results(args.results_dir)

    if df.empty:
        print("No results found.")
        return

    print(f"Found {len(df)} runs.\n")

    # Full results
    df.to_csv(args.output.replace(".csv", "_full.csv"), index=False)

    # Summary table
    summary = summary_table(df)
    summary.to_csv(args.output, index=False)

    print("Summary (mean +/- std mAP across seeds):\n")
    print(summary.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
