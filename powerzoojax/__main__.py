"""PowerZooJax CLI — ``python -m powerzoojax``.

Usage::

    # List available presets
    python -m powerzoojax --list-presets

    # Train with a preset
    python -m powerzoojax --preset case5-economic-dispatch --seed 42

    # Train with a YAML config (a preset is required to supply the env;
    # the YAML overrides training hyperparameters on top of the preset)
    python -m powerzoojax --preset case5-economic-dispatch --config experiment.yaml --output result.json

    # Override config fields inline
    python -m powerzoojax --preset battery-soc-tracking --total_timesteps 500000

    # Save result JSON
    python -m powerzoojax --preset case5-economic-dispatch --output result.json
"""

import argparse
import json
import sys


def _parse_dot_overrides(unknown: list) -> dict:
    """Parse ``--key value`` pairs from unknown args.

    Supports int, float, and string coercion.  Boolean flags are not supported
    (use YAML config for complex settings).
    """
    overrides = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if tok.startswith("--") and i + 1 < len(unknown):
            key = tok[2:].replace("-", "_")
            val_str = unknown[i + 1]
            # Coerce to numeric types when possible
            try:
                val = int(val_str)
            except ValueError:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str
            overrides[key] = val
            i += 2
        else:
            i += 1
    return overrides


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m powerzoojax",
        description="PowerZooJax RL training CLI.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        metavar="NAME",
        help="Preset name (see --list-presets for available options).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a YAML training config file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="PATH",
        help="Save TrainResult summary JSON to this path.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available presets and exit.",
    )

    args, unknown = parser.parse_known_args(argv)

    if args.list_presets:
        from powerzoojax.rl.presets import list_presets
        presets = list_presets()
        print(f"\n{'Preset':<35} {'Algo':<18} {'Steps':>10}  Description")
        print("-" * 90)
        for p in presets:
            print(
                f"  {p['name']:<33} {p['algo']:<18} {p['total_timesteps']:>10}"
                f"  {p['description']}"
            )
        print()
        return 0

    if args.preset is None:
        parser.print_help()
        print("\nError: must provide --preset (--config is optional and only overrides hyperparameters).", file=sys.stderr)
        return 1

    # Parse inline overrides (e.g. --total_timesteps 500000 --num_envs 16)
    overrides = _parse_dot_overrides(unknown)
    overrides.pop("seed", None)  # seed is handled by --seed

    # Load config from YAML if provided
    config = None
    if args.config:
        from powerzoojax.rl.config import load_config
        config = load_config(args.config)

    from powerzoojax.rl.train import train
    result = train(
        args.preset if args.preset else config,
        config=config if args.config else None,
        seed=args.seed,
        **overrides,
    )

    print(json.dumps(result.summary, indent=2, default=str))

    if args.output:
        result.save(args.output)
        print(f"\nResult saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
