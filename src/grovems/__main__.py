"""grovems CLI: Oktoberfest/Percolator rescoring + PSA + SUOD IForest scoring."""

from __future__ import annotations

import argparse
from pathlib import Path

from grovems import __version__, runner
from grovems.config import GrovemsConfig
from grovems.utils import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="grovems", description=__doc__)
    parser.add_argument("-v", "--version", action="version", version=__version__)

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the pipeline")
    run_parser.add_argument("--config", required=True, help="Path to a config YAML file")
    run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config value; may be given multiple times",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = GrovemsConfig.from_yaml(args.config)
    config.apply_overrides(args.overrides)

    outdir = Path(config.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_file=outdir / "grovems.log")

    runner.run(config)


if __name__ == "__main__":
    main()
