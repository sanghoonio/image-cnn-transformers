"""Generate iteratively stratified training subsets."""

import argparse
import json

from voc_bench.data.stratified_split import create_subsets
from voc_bench.data.voc_dataset import VOC_CLASSES


def main():
    parser = argparse.ArgumentParser(
        description="Generate stratified multi-label subsets of VOC training data"
    )
    parser.add_argument(
        "--data-root", type=str, default="data/VOCdevkit",
        help="Path to VOCdevkit directory",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/subsets",
        help="Directory to save subset index files",
    )
    parser.add_argument(
        "--output-stats", type=str, default=None,
        help="Path to save subset statistics JSON (optional)",
    )
    args = parser.parse_args()

    print("Generating stratified subsets...")
    results = create_subsets(args.data_root, args.output_dir)

    print(f"\nSaved subset indices to {args.output_dir}/\n")

    # Print summary
    for (fraction, seed), info in sorted(results.items()):
        min_count = info["min_class_count"]
        status = "OK" if min_count >= 5 else f"WARNING: min class count = {min_count}"
        print(
            f"  frac={fraction:.2f} seed={seed}: "
            f"{info['n_images']} images, "
            f"min class count={min_count} [{status}]"
        )

    # Print detailed per-class counts for 5% subset
    print("\nPer-class counts for 5% subset (seed=42):")
    key = (0.05, 42)
    if key in results:
        counts = results[key]["per_class_counts"]
        for cls_name, count in zip(VOC_CLASSES, counts):
            flag = " <-- WARNING" if count < 5 else ""
            print(f"  {cls_name:15s}  {count:4d}{flag}")

    if args.output_stats:
        # Convert tuple keys to strings for JSON
        json_results = {
            f"frac{f:.2f}_seed{s}": v for (f, s), v in results.items()
        }
        with open(args.output_stats, "w") as fp:
            json.dump(json_results, fp, indent=2)
        print(f"\nStats saved to {args.output_stats}")


if __name__ == "__main__":
    main()
