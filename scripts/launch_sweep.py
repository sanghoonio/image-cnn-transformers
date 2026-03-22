"""Generate per-run configs and SLURM job array for the full experiment sweep."""

import argparse
from pathlib import Path

import yaml

from voc_bench.config import load_config, save_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate configs and SLURM job array for experiment sweep"
    )
    parser.add_argument(
        "--sweep-config", type=str, default="configs/sweep.yaml",
        help="Path to sweep YAML",
    )
    parser.add_argument(
        "--base-config", type=str, default="configs/base.yaml",
        help="Path to base config YAML",
    )
    parser.add_argument(
        "--output-dir", type=str, default="configs/generated",
        help="Directory to write per-run config files",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs without writing files",
    )
    args = parser.parse_args()

    with open(args.sweep_config) as f:
        sweep = yaml.safe_load(f)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_paths = []

    for model_name in sweep["models"]:
        model_config_path = f"configs/{model_name}.yaml"
        for fraction in sweep["fractions"]:
            for seed in sweep["seeds"]:
                overrides = {
                    "fraction": fraction,
                    "seed": seed,
                    "augmentation": sweep["augmentation"],
                }

                cfg = load_config(
                    args.base_config, model_config_path, overrides=overrides
                )

                pretrain_tag = "pretrained" if cfg.pretrained else "scratch"
                run_name = (
                    f"{cfg.model_name}_{pretrain_tag}_"
                    f"frac{fraction:.2f}_seed{seed}"
                )
                config_path = out_dir / f"{run_name}.yaml"

                if args.dry_run:
                    print(f"  {run_name}")
                else:
                    save_config(cfg, config_path)
                    config_paths.append(str(config_path))

    if args.dry_run:
        print(f"\n{len(sweep['models']) * len(sweep['fractions']) * len(sweep['seeds'])} "
              f"configurations (dry run)")
        return

    # Write manifest file
    manifest_path = out_dir / "sweep_configs.txt"
    with open(manifest_path, "w") as f:
        for p in config_paths:
            f.write(p + "\n")

    print(f"Generated {len(config_paths)} configs in {out_dir}/")
    print(f"Manifest: {manifest_path}")
    print(f"\nTo submit on Rivanna:")
    print(f"  sbatch --array=0-{len(config_paths)-1}%20 slurm/sweep_array.sh")


if __name__ == "__main__":
    main()
