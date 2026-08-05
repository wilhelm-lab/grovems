from __future__ import annotations

import codecs
import logging
import sys

import pandas as pd

from ..utils import configure_logging

logger = logging.getLogger(__name__)


def drop_columns(input_file: str, output_file: str, delimiter: str, columns_to_drop: list[str]) -> None:
    """Read a delimited file, drop the given columns, and write the result.

    Args:
        input_file: Path to the input delimited file.
        output_file: Path to write the column-dropped file to.
        delimiter: Field delimiter, with escape sequences already decoded (e.g. a real
            tab character, not the two-character string ``"\\t"``).
        columns_to_drop: Column names to drop; names not present are ignored.
    """
    logger.info("Reading from '%s'", input_file)
    df = pd.read_csv(input_file, sep=delimiter, engine="c")

    logger.info("Dropping columns: %s", columns_to_drop)
    df = df.drop(columns=columns_to_drop, errors="ignore")

    logger.info("Writing to '%s'", output_file)
    df.to_csv(output_file, sep=delimiter, index=False)

    logger.info("Wrote %d rows, %d columns to '%s'", len(df), len(df.columns), output_file)


def main() -> None:
    configure_logging()

    if len(sys.argv) < 5:
        logger.error("Usage: python drop_columns.py <input> <output> <delimiter> <col1> [<col2> ...]")
        logger.error("Example: python drop_columns.py data.csv cleaned.csv ',' age address")
        logger.error(r"         python drop_columns.py data.tsv cleaned.tsv '\t' col1 col2")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    delimiter = codecs.decode(sys.argv[3], "unicode_escape")
    columns_to_drop = sys.argv[4:]

    drop_columns(input_file, output_file, delimiter, columns_to_drop)


if __name__ == "__main__":
    main()
