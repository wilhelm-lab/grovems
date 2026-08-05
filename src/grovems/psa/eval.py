from __future__ import annotations

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from oktoberfest.utils.config import Config

from ..utils import configure_logging

logger = logging.getLogger(__name__)

DENOVO_COLUMNS = [
    "RAW_FILE",
    "SCAN_NUMBER",
    "MODIFIED_SEQUENCE",
    "PRECURSOR_CHARGE",
    "MASS",
    "SCORE",
    "REVERSE",
    "SEQUENCE",
    # "PEPTIDE_LENGTH",
    # "RETENTION_TIME",
    # "COLLISION_ENERGY",
    # "MOST_INTENSE_PEAK",
    "AA_SCORE",
]

DATABASE_COLUMNS = [
    "RAW_FILE",
    "SCAN_NUMBER",
    "MODIFIED_SEQUENCE",
    "PRECURSOR_CHARGE",
    "MASS",
    "SCORE",
    "SEQUENCE",
    "num_missed_cleavages",
    "REVERSE",
    # "PEPTIDE_LENGTH",
    # "retention_time_sec",
    # "COLLISION_ENERGY",
    # "MOST_INTENSE_PEAK",
]

SPECID_COLUMNS = ["RAW_FILE", "SCAN_NUMBER", "MODIFIED_SEQUENCE", "PRECURSOR_CHARGE"]
PERCOLATOR_COLUMNS = ["PSMId", "score", "q-value", "posterior_error_prob"]
INPUT_DATASETS = (
    "DENOVO_SEARCH",
    "DATABASE_SEARCH",
    "DENOVO_PIN",
    "DATABASE_PIN",
    "DENOVO_PERCOLATOR",
    "DENOVO_DECOY_PERCOLATOR",
    "DATABASE_PERCOLATOR",
    "DATABASE_DECOY_PERCOLATOR",
)


@dataclass
class DataHolder:
    # Raw data
    DENOVO_SEARCH: Optional[pd.DataFrame] = None
    DATABASE_SEARCH: Optional[pd.DataFrame] = None
    # Percolator input
    DENOVO_PIN: Optional[pd.DataFrame] = None
    DATABASE_PIN: Optional[pd.DataFrame] = None
    # Percolator output
    DENOVO_PERCOLATOR: Optional[pd.DataFrame] = None
    DENOVO_DECOY_PERCOLATOR: Optional[pd.DataFrame] = None
    DATABASE_PERCOLATOR: Optional[pd.DataFrame] = None
    DATABASE_DECOY_PERCOLATOR: Optional[pd.DataFrame] = None
    # Merged outputs
    MERGED_DENOVO: Optional[pd.DataFrame] = None
    MERGED_DATABASE: Optional[pd.DataFrame] = None
    MERGED_DF: Optional[pd.DataFrame] = None
    MERGED_PSM: Optional[pd.DataFrame] = None
    MERGED_SCAN: Optional[pd.DataFrame] = None
    # Iforest output
    IFOREST_RESULTS: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class LoadSpec:
    path_key: str
    dataname: str
    sep: Optional[str] = ","
    usecols: Optional[list[str]] = None


class ResultLoader:
    """Load, cache, and merge rescoring pipeline result datasets."""

    def __init__(
        self,
        database_oktoberfest_config: Union[Config, Path],
        denovo_oktoberfest_config: Union[Config, Path],
        database_pin_name: str = "rescore.tab",
        denovo_pin_name: str = "rescore.tab",
        max_workers: Optional[int] = None,
    ):
        logger.info("Initializing ResultLoader")

        self.denovo_cfg = self._load_config(denovo_oktoberfest_config)
        self.database_cfg = self._load_config(database_oktoberfest_config)

        self.database_pin_name = database_pin_name
        self.denovo_pin_name = denovo_pin_name
        self.max_workers = max_workers

        self.path_mapper: dict[str, Optional[Path]] = {}
        self._load_paths()

        self.data: DataHolder = DataHolder()
        logger.info("ResultLoader ready with max_workers=%s", self.max_workers or "auto")

    @staticmethod
    def _load_config(cfg: Union[Config, Path]) -> Config:
        if isinstance(cfg, Config):
            return cfg

        config = Config()
        config.read(cfg)
        logger.info("Loaded configuration from %s", cfg)
        return config

    def set_path(self, path_name: str, path_location: Union[Path, str, None]) -> None:
        self.path_mapper[path_name.lower()] = None if path_location is None else Path(path_location)
        logger.info(
            "Set path %s -> %s",
            path_name.lower(),
            self.path_mapper[path_name.lower()],
        )

    def _load_paths(self) -> None:
        """Load all paths derived from the configs."""
        logger.info("Loading paths")

        percolator_dir = Path("results") / "percolator"
        target_name = "rescore.percolator.psms.txt"
        decoy_name = "rescore.percolator.decoy.psms.txt"

        denovo_out = Path(self.denovo_cfg.output)
        database_out = Path(self.database_cfg.output)

        denovo_base = denovo_out / percolator_dir
        database_base = database_out / percolator_dir

        self.set_path("denovo_pin", denovo_base / self.denovo_pin_name)
        self.set_path("database_pin", database_base / self.database_pin_name)

        self.set_path("denovo_search", denovo_out / "msms")
        self.set_path("database_search", database_out / "msms")

        self.set_path("denovo_percolator_target", denovo_base / target_name)
        self.set_path("denovo_percolator_decoy", denovo_base / decoy_name)
        self.set_path("database_percolator_target", database_base / target_name)
        self.set_path("database_percolator_decoy", database_base / decoy_name)
        logger.info("Configured %d input paths", len(self.path_mapper))

    @staticmethod
    def _log_frame_state(name: str, df: pd.DataFrame) -> None:
        logger.info("Loaded %s with %d rows and %d columns", name, len(df), len(df.columns))
        logger.debug("%s columns: %s", name, list(df.columns))

    @staticmethod
    def reader(
        path: Path,
        *,
        sep: Optional[str] = ",",
        usecols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Read file into a DataFrame."""
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.is_dir():
            rescore_files = sorted(path.glob("*.rescore"))
            if not rescore_files:
                raise ValueError(f"No .rescore files found in directory: {path}")
            logger.info("Reading %d .rescore files from %s", len(rescore_files), path)
            if len(rescore_files) == 1:
                return pd.read_csv(rescore_files[0], sep=sep, usecols=usecols)

            worker_count = ResultLoader._default_worker_count(len(rescore_files))
            logger.info("Using %d worker(s) to read directory input %s", worker_count, path)
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                frames = list(
                    executor.map(
                        ResultLoader._read_tabular_file,
                        rescore_files,
                        [sep] * len(rescore_files),
                        [usecols] * len(rescore_files),
                    )
                )
            return pd.concat(
                frames,
                ignore_index=True,
                copy=False,
            )

        suffix = path.suffix.lower()
        logger.info("Reading %s file from %s", suffix or "tabular", path)

        if suffix == ".parquet":
            return pd.read_parquet(path, columns=usecols)
        if suffix in {".csv", ".tsv", ".txt", ".tab"}:
            return pd.read_csv(path, sep=sep, usecols=usecols)

        raise ValueError(f"Unsupported file type: {suffix} (path={path})")

    @staticmethod
    def _read_tabular_file(
        path: Path,
        sep: Optional[str],
        usecols: Optional[list[str]],
    ) -> pd.DataFrame:
        return pd.read_csv(path, sep=sep, usecols=usecols)

    @staticmethod
    def _default_worker_count(task_count: int) -> int:
        if task_count <= 1:
            return 1
        return min(8, task_count)

    def _worker_count(self, task_count: int) -> int:
        if task_count <= 1:
            return 1
        if self.max_workers is not None:
            return max(1, min(self.max_workers, task_count))
        return self._default_worker_count(task_count)

    def _read_spec(self, spec: LoadSpec) -> pd.DataFrame:
        path = self.path_mapper.get(spec.path_key.lower())
        if path is None:
            raise KeyError(f"Path key not found or unset: {spec.path_key.lower()}")

        try:
            df = self.reader(path, sep=spec.sep, usecols=spec.usecols)
            self._log_frame_state(spec.dataname, df)
            return df
        except FileNotFoundError:
            logger.warning("File not found for %s (%s). Using empty DataFrame.", spec.dataname, path)
            return pd.DataFrame()

    def _load_specs_parallel(
        self,
        specs: list[LoadSpec],
    ) -> dict[str, pd.DataFrame]:
        if not specs:
            return {}

        worker_count = self._worker_count(len(specs))
        logger.info("Loading %d dataset(s) with %d worker(s)", len(specs), worker_count)
        if worker_count == 1:
            loaded = {spec.dataname: self._read_spec(spec) for spec in specs}
        else:
            loaded: dict[str, pd.DataFrame] = {}
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {executor.submit(self._read_spec, spec): spec for spec in specs}
                for future in as_completed(future_map):
                    spec = future_map[future]
                    loaded[spec.dataname] = future.result()

        return loaded

    def clear_data(self, *datanames: str) -> None:
        if datanames:
            logger.info("Clearing cached datasets: %s", ", ".join(datanames))
        for dataname in datanames:
            if hasattr(self.data, dataname):
                setattr(self.data, dataname, None)

    def _make_specid(self, df: pd.DataFrame) -> pd.Series:
        missing = [c for c in SPECID_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns needed to build SpecId: {missing}")
        specid = df[SPECID_COLUMNS[0]].astype(str)
        for column in SPECID_COLUMNS[1:]:
            specid = specid.str.cat(df[column].astype(str), sep="-")
        return specid

    def _prepare_search_frame(
        self,
        df: pd.DataFrame,
        keep_columns: list[str],
    ) -> pd.DataFrame:
        keep = [column for column in keep_columns if column in df.columns]
        search = df.loc[:, keep].copy()
        search["SpecId"] = self._make_specid(search)
        logger.info("Prepared search frame with %d rows and %d columns", len(search), len(search.columns))
        return search

    @staticmethod
    def _merge_label_mapping() -> dict[str, str]:
        return {
            "right_only": "denovo_only",
            "left_only": "database_only",
            "both": "shared",
        }

    @staticmethod
    def _coalesce_merged_column(merged: pd.DataFrame, column: str) -> None:
        left_column = f"{column}_database"
        right_column = f"{column}_denovo"
        if left_column not in merged.columns and right_column not in merged.columns:
            return

        left_values = (
            merged[left_column] if left_column in merged.columns else pd.Series(index=merged.index, dtype="object")
        )
        right_values = (
            merged[right_column] if right_column in merged.columns else pd.Series(index=merged.index, dtype="object")
        )
        merged[column] = left_values.combine_first(right_values)

    @staticmethod
    def _chimeric_scan_keys(merged_database: pd.DataFrame) -> set[tuple[str, int]]:
        required = ["RAW_FILE", "SCAN_NUMBER", "SEQUENCE"]
        missing = [column for column in required if column not in merged_database.columns]
        if missing or merged_database.empty:
            return set()

        database = merged_database.dropna(subset=required).copy()
        if database.empty:
            return set()

        database["SCAN_NUMBER"] = pd.to_numeric(database["SCAN_NUMBER"], errors="coerce")
        database = database.dropna(subset=["SCAN_NUMBER"])
        if database.empty:
            return set()

        counts = database.groupby(["RAW_FILE", "SCAN_NUMBER"], observed=True)["SEQUENCE"].nunique()
        chimeric = counts[counts > 1]
        return {(str(raw_file), int(scan_number)) for raw_file, scan_number in chimeric.index}

    @staticmethod
    def _add_chimeric_column(merged: pd.DataFrame, chimeric_scan_keys: set[tuple[str, int]]) -> None:
        if not {"RAW_FILE", "SCAN_NUMBER"}.issubset(merged.columns):
            merged["chimeric"] = False
            return

        if not chimeric_scan_keys:
            merged["chimeric"] = False
            return

        scan_number = pd.to_numeric(merged["SCAN_NUMBER"], errors="coerce")
        keys = pd.Series(
            zip(merged["RAW_FILE"].astype(str), scan_number),
            index=merged.index,
            dtype="object",
        )
        merged["chimeric"] = keys.map(
            lambda key: pd.notna(key[1]) and (key[0], int(key[1])) in chimeric_scan_keys
        )

    @staticmethod
    def _add_sequence_match_column(merged: pd.DataFrame) -> None:
        sequence_columns = ["SEQUENCE_database", "SEQUENCE_denovo"]
        if not set(sequence_columns).issubset(merged.columns):
            merged["unmodified_sequence_match"] = False
            merged["modified_sequence_match"] = False
            merged["precursor_charge_match"] = False
            merged["sequence_match"] = False
            return

        merged["unmodified_sequence_match"] = (
            merged["SEQUENCE_database"].notna()
            & merged["SEQUENCE_denovo"].notna()
            & merged["SEQUENCE_database"].eq(merged["SEQUENCE_denovo"])
        )

        modified_columns = ["MODIFIED_SEQUENCE_database", "MODIFIED_SEQUENCE_denovo"]
        if set(modified_columns).issubset(merged.columns):
            merged["modified_sequence_match"] = (
                merged["MODIFIED_SEQUENCE_database"].notna()
                & merged["MODIFIED_SEQUENCE_denovo"].notna()
                & merged["MODIFIED_SEQUENCE_database"].eq(merged["MODIFIED_SEQUENCE_denovo"])
            )
        else:
            merged["modified_sequence_match"] = False

        charge_columns = ["PRECURSOR_CHARGE_database", "PRECURSOR_CHARGE_denovo"]
        if set(charge_columns).issubset(merged.columns):
            database_charge = pd.to_numeric(merged["PRECURSOR_CHARGE_database"], errors="coerce")
            denovo_charge = pd.to_numeric(merged["PRECURSOR_CHARGE_denovo"], errors="coerce")
            merged["precursor_charge_match"] = database_charge.notna() & denovo_charge.notna() & database_charge.eq(
                denovo_charge
            )
        else:
            merged["precursor_charge_match"] = False

        merged["sequence_match"] = merged["modified_sequence_match"] & merged["precursor_charge_match"]

    def _merge_search_results(
        self,
        merged_database: pd.DataFrame,
        merged_denovo: pd.DataFrame,
        *,
        left_on: list[str],
        right_on: list[str],
        chimeric_scan_keys: Optional[set[tuple[str, int]]] = None,
        add_sequence_match: bool = False,
    ) -> pd.DataFrame:
        logger.info("Merging database and denovo results on %s", ", ".join(left_on))
        merged = merged_database.merge(
            merged_denovo,
            how="outer",
            left_on=left_on,
            right_on=right_on,
            indicator=True,
            suffixes=("_database", "_denovo"),
        )

        for column in ("SpecId", "RAW_FILE", "SCAN_NUMBER"):
            self._coalesce_merged_column(merged, column)

        if chimeric_scan_keys is not None:
            self._add_chimeric_column(merged, chimeric_scan_keys)
        if add_sequence_match:
            self._add_sequence_match_column(merged)

        merged["_merge"] = merged["_merge"].map(self._merge_label_mapping())
        logger.info("Merged frame has %d rows and %d columns", len(merged), len(merged.columns))
        return merged

    @staticmethod
    def _write_partitioned_parquet(df: pd.DataFrame, output_path: Path) -> None:
        logger.info(
            "Writing %d rows and %d columns to %s partitioned by RAW_FILE",
            len(df),
            len(df.columns),
            output_path,
        )
        df.to_parquet(
            output_path,
            index=False,
            engine="pyarrow",
            partition_cols=["RAW_FILE"],
        )

    def _load_percolator_scores(
        self,
        target_key: str,
        decoy_key: str,
        cache_prefix: str,
    ) -> pd.Series:
        specs = [
            LoadSpec(target_key, f"{cache_prefix}_PERCOLATOR", sep="\t", usecols=PERCOLATOR_COLUMNS),
            LoadSpec(decoy_key, f"{cache_prefix}_DECOY_PERCOLATOR", sep="\t", usecols=PERCOLATOR_COLUMNS),
        ]
        loaded_frames = self._load_specs_parallel(specs)
        frames = []
        for spec in specs:
            frame = loaded_frames.get(spec.dataname)
            if frame is None or frame.empty:
                logger.warning("No percolator rows found for %s", spec.dataname)
                continue

            missing = [column for column in PERCOLATOR_COLUMNS if column not in frame.columns]
            if missing:
                raise KeyError(f"Missing percolator columns in {spec.path_key}: {missing}")
            frames.append(frame.set_index("PSMId")["score"])

        if not frames:
            logger.warning("No percolator scores available for %s", cache_prefix)
            return pd.Series(dtype="float64")

        scores = pd.concat(frames, copy=False)
        deduplicated = scores[~scores.index.duplicated(keep="last")]
        logger.info("Loaded %d unique percolator scores for %s", len(deduplicated), cache_prefix)
        return deduplicated

    def _load_merge_inputs(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        specs = [
            LoadSpec("denovo_pin", "DENOVO_PIN", sep="\t"),
            LoadSpec("database_pin", "DATABASE_PIN", sep="\t"),
            LoadSpec("denovo_search", "DENOVO_SEARCH", sep=",", usecols=DENOVO_COLUMNS),
            LoadSpec("database_search", "DATABASE_SEARCH", sep=",", usecols=DATABASE_COLUMNS),
        ]
        loaded = self._load_specs_parallel(specs)
        denovo_pin = loaded.get("DENOVO_PIN")
        database_pin = loaded.get("DATABASE_PIN")
        denovo_search = loaded.get("DENOVO_SEARCH")
        database_search = loaded.get("DATABASE_SEARCH")

        if denovo_pin is None or database_pin is None or denovo_search is None or database_search is None:
            raise RuntimeError("Some required inputs were not loaded (PIN/search).")

        logger.info(
            "Merge inputs ready: denovo_pin=%d, database_pin=%d, denovo_search=%d, database_search=%d",
            len(denovo_pin),
            len(database_pin),
            len(denovo_search),
            len(database_search),
        )
        return denovo_pin, database_pin, denovo_search, database_search

    def merge_results(self) -> pd.DataFrame:
        """
        Merge search outputs + PIN + percolator scores for denovo and database.
        Produces:
          - self.data.MERGED_DENOVO
          - self.data.MERGED_DATABASE
          - self.data.MERGED_PSM (outer merge on SpecId)
          - self.data.MERGED_SCAN (outer merge on RAW_FILE + SCAN_NUMBER)
        """
        logger.info("Starting merge_results")
        denovo_pin, database_pin, denovo_search, database_search = self._load_merge_inputs()

        denovo_pin["percolator_score"] = denovo_pin["SpecId"].map(
            self._load_percolator_scores(
                "denovo_percolator_target",
                "denovo_percolator_decoy",
                "DENOVO",
            )
        )
        database_pin["percolator_score"] = database_pin["SpecId"].map(
            self._load_percolator_scores(
                "database_percolator_target",
                "database_percolator_decoy",
                "DATABASE",
            )
        )

        denovo_search = self._prepare_search_frame(denovo_search, DENOVO_COLUMNS)
        database_search = self._prepare_search_frame(database_search, DATABASE_COLUMNS)

        # --- Merge search + pin ---
        merged_denovo = denovo_search.merge(
            denovo_pin,
            how="outer",
            on="SpecId",
            indicator=False,
        )
        merged_database = database_search.merge(
            database_pin,
            how="outer",
            on="SpecId",
            indicator=False,
        )

        self.data.MERGED_DENOVO = merged_denovo
        self.data.MERGED_DATABASE = merged_database

        chimeric_scan_keys = self._chimeric_scan_keys(merged_database)
        merged_psm = self._merge_search_results(
            merged_database,
            merged_denovo,
            left_on=["SpecId"],
            right_on=["SpecId"],
            chimeric_scan_keys=chimeric_scan_keys,
        )
        merged_scan = self._merge_search_results(
            merged_database,
            merged_denovo,
            left_on=["RAW_FILE", "SCAN_NUMBER"],
            right_on=["RAW_FILE", "SCAN_NUMBER"],
            chimeric_scan_keys=chimeric_scan_keys,
            add_sequence_match=True,
        )

        self.data.MERGED_PSM = merged_psm
        self.data.MERGED_SCAN = merged_scan
        self.data.MERGED_DF = merged_psm

        logger.info("Merged database results: %d rows", len(merged_database))
        logger.info("Merged denovo results: %d rows", len(merged_denovo))
        logger.info("Merged PSM results: %d rows", len(merged_psm))
        logger.info("Merged scan results: %d rows", len(merged_scan))
        return merged_psm

    def write_merged(self, output_path: Path = Path("./all_in_one")) -> None:
        output_path = Path(output_path)
        logger.info("Writing merged outputs under %s", output_path)
        self.merge_results()
        output_path.mkdir(parents=True, exist_ok=True)

        outputs = {
            "merged_database": self.data.MERGED_DATABASE,
            "merged_denovo": self.data.MERGED_DENOVO,
            "merged_PSM": self.data.MERGED_PSM,
            "merged_SCAN": self.data.MERGED_SCAN,
        }
        for name, df in outputs.items():
            if df is None:
                logger.warning("Skipping %s because no dataframe was produced", name)
                continue
            self._write_partitioned_parquet(df, output_path / name)

        self.clear_data(*INPUT_DATASETS)
        logger.info("Finished writing merged outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a rescoring run's database/de novo Percolator outputs and write merged parquet."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Rescoring run directory containing rescoring_config_database.json/rescoring_config_denovo.json",
    )
    parser.add_argument("--output", required=True, help="Output directory for the merged parquet datasets")
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    res = ResultLoader(
        database_oktoberfest_config=Path(args.input) / "rescoring_config_database.json",
        denovo_oktoberfest_config=Path(args.input) / "rescoring_config_denovo.json",
        denovo_pin_name="rescore.filtered.tab",
        database_pin_name="rescore.filtered.tab",
    )
    res.write_merged(Path(args.output))


if __name__ == "__main__":
    main()
