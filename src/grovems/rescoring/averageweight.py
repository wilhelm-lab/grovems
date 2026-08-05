from __future__ import annotations

import argparse
import logging

import pandas as pd

from ..utils import configure_logging

logger = logging.getLogger(__name__)


def parse_and_write_avg_weights(
    path: str,
    out_path: str,
    decimals: int = 4,
    return_dataframe: bool = False,
) -> pd.DataFrame | None:
    """Parse per-bin weights from ``path``, average them, and write the result to ``out_path``.

    Args:
        path: Input file with repeated (header, normalized weights, raw weights)
            line triples, one triple per bin.
        out_path: Output file to write the averaged weights to.
        decimals: Decimal places to round the averages to.
        return_dataframe: If ``True``, also return a DataFrame of the averages.

    Returns:
        A DataFrame with ``feature``/``average_normalized``/``average_raw`` columns if
        ``return_dataframe`` is ``True``, else ``None``.
    """
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    i = 0
    header: list[str] = []
    bins_norm: list[list[float]] = []
    bins_raw: list[list[float]] = []
    while i < len(lines):
        header = lines[i].split("\t")
        norm = list(map(float, lines[i + 1].split("\t")))
        raw = list(map(float, lines[i + 2].split("\t")))
        bins_norm.append(norm)
        bins_raw.append(raw)
        i += 3

    logger.info("Averaging %d bin(s) of weights from '%s'", len(bins_norm), path)
    avg_norm = [sum(x) / len(bins_norm) for x in zip(*bins_norm)]
    avg_raw = [sum(x) / len(bins_raw) for x in zip(*bins_raw)]

    with open(out_path, "w") as f:
        f.write("# Average weights across all bins\n")
        f.write("\t".join(header) + "\n")
        f.write("\t".join(f"{x:.{decimals}f}" for x in avg_norm) + "\n")
        f.write("\t".join(f"{x:.{decimals}f}" for x in avg_raw) + "\n")

    logger.info("Averages written to '%s'", out_path)

    if return_dataframe:
        return pd.DataFrame(
            {
                "feature": header,
                "average_normalized": avg_norm,
                "average_raw": avg_raw,
            }
        )
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute average weights across bins.")
    parser.add_argument("--input", required=True, help="Input file path containing bin weights")
    parser.add_argument("--output", required=True, help="Output file path for averaged weights")
    parser.add_argument("--decimals", type=int, default=4, help="Decimal precision for averages")
    parser.add_argument("--return_dataframe", action="store_true", help="Log the averaged DataFrame")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    df = parse_and_write_avg_weights(
        args.input,
        args.output,
        args.decimals,
        return_dataframe=args.return_dataframe,
    )
    if df is not None:
        logger.info("Averaged weights:\n%s", df)


if __name__ == "__main__":
    main()
