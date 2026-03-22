"""Download PASCAL VOC 2012 and verify dataset statistics."""

import argparse
import json
from pathlib import Path

import torchvision

from voc_bench.data.stats import compute_dataset_stats, print_stats


def main():
    parser = argparse.ArgumentParser(description="Download and verify PASCAL VOC 2012")
    parser.add_argument(
        "--data-root", type=str, default="data/VOCdevkit",
        help="Path to VOCdevkit directory",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download the dataset if not present",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save stats JSON (optional)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.download:
        # torchvision downloads to {root}/VOCdevkit/VOC2012
        # We pass the parent so it creates VOCdevkit inside data_root's parent
        download_root = data_root.parent if data_root.name == "VOCdevkit" else data_root
        print(f"Downloading VOC 2012 to {download_root}...")
        torchvision.datasets.VOCDetection(
            root=str(download_root), year="2012", image_set="train", download=True
        )
        print("Download complete.")

    voc_path = data_root / "VOC2012"
    if not voc_path.exists():
        print(f"Error: {voc_path} not found. Run with --download to fetch the dataset.")
        return

    print("Computing dataset statistics...\n")
    stats = compute_dataset_stats(str(data_root))
    print_stats(stats)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nStats saved to {output_path}")


if __name__ == "__main__":
    main()
