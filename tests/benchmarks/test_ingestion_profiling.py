"""
Profile ROOT/parquet ingestion setup and read cost across five strategies:

- ``root_dask``: `Ingestor(format="root")` -- `uproot.dask` graph/form construction.
- ``root_iterative``: `IterativeIngestor` -- `uproot.num_entries` metadata only, no dask graph.
- ``parquet_dask``: `Ingestor(format="parquet")` -- `dak.from_parquet` graph construction.
- ``parquet_iterative``: `IterativeParquetIngestor` -- `pyarrow.parquet.ParquetFile` metadata only,
  no dask graph.
- ``rdataloader``: `ROOTPaddedDataModule` -- ROOT's native RDataLoader, reading the same .root
  files as the two uproot strategies. Skipped automatically when PyROOT is unavailable.

Note that ``rdataloader`` is not a like-for-like comparison: the other four yield jagged awkward
arrays, while RDataLoader yields *fixed-width padded* float32 torch tensors. For this column set
that is ~2042 padded values per event against ~780 real ones, plus a tensor conversion the others
never pay. Read it as "what a training-ready batch costs", not "what reading these columns costs".

See `tests/benchmarks/plot_ingestion_profiling.py` for the plots built from this data, and
`tests/benchmarks/test_root_vs_parquet.py` for the original, broader-scope ROOT-vs-parquet
benchmark this suite complements.

`tests/benchmarks/conftest.py` writes the result to `tests/benchmarks/results/ingestion_profiling.json`
automatically

Requires the `DELPHES_DATA_ROOT` and `DELPHES_DATA_PARQUET` environment variables (see
`tests/conftest.py`). Files are large (~950 MB / 10k events / ~800 branches each for ROOT) --
mind `num_files` when adding new parametrizations.

Disclaimer: Part of this code was written with the help of GPT-5 and Claude Sonnet 5.
"""

from typing import Any, Callable, List

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from needle.etl.array import resolve_paths
from needle.etl.dask_ingestor import Ingestor
from needle.etl.iterative_ingestor import IterableIngestor

pytestmark = pytest.mark.benchmark

# Columns known to be valid for the Delphes v1 dataset, see tests/conf_tests/datasets/delphes.yaml.
# ("Track.*"/"Photon.*" only -- avoids the invalid branches dropped elsewhere, e.g. "fBits".)
COLUMN_SETS: dict[str, List[str]] = {
    "few": ["Track.PT"],
    "many": [
        "Track.PID",
        "Track.Charge",
        "Track.P",
        "Track.PT",
        "Track.Eta",
        "Track.Phi",
        "Track.C",
        "Track.Mass",
        "Photon.PT",
        "Photon.Eta",
        "Photon.Phi",
        "Photon.E",
        "Photon.T",
        "Photon.EhadOverEem",
    ],
}

NUM_FILES = [
    pytest.param(1, id="files_1", marks=pytest.mark.slow),
    pytest.param(2, id="files_2", marks=pytest.mark.slow),
    pytest.param(3, id="files_3", marks=pytest.mark.slow),
    pytest.param(4, id="files_4", marks=pytest.mark.slow),
    pytest.param(5, id="files_5", marks=pytest.mark.slow),
    pytest.param(7, id="files_7", marks=pytest.mark.slow),
    pytest.param(10, id="files_10", marks=pytest.mark.slow),
    pytest.param(15, id="files_15", marks=pytest.mark.slow),
    pytest.param(20, id="files_20", marks=pytest.mark.slow),
]

NUM_ROUNDS: int = 3
NUM_ITERATIONS: int = 1
NUM_WARMUP_ROUNDS: int = 0

# --- RDataLoader ------------------------------------------------------------------------------
# `ROOTPaddedDataModule` wraps ROOT's native RDataLoader. It reads the *same* .root files as the
# two uproot strategies, so any difference is attributable to the reader rather than the format.
#
# Unlike uproot/awkward, RDataLoader materialises jagged columns at a *fixed* width, so every
# collection needs an explicit cap. These are the global maxima of the `Track_size` / `Photon_size`
# counter branches over all 21 files -- anything smaller would silently truncate events and make
# the comparison meaningless. Note this costs RDataLoader real work: Track averages 97 entries per
# event against a 253-wide buffer, and Photon averages 0.1 against 3.
MAX_VEC_SIZES: dict[str, int] = {"Track": 253, "Photon": 3}

TREE_NAME: str = "Delphes"
RDATALOADER_BATCH_SIZE: int = 1024
RDATALOADER_BATCHES_IN_MEMORY: int = 10


def _filter_name(columns: List[str]) -> Callable[[str], bool]:
    """Build a `filter_name` predicate that keeps only the requested columns (ROOT only)."""

    def _filter(name: str) -> bool:
        return name in columns

    return _filter


def _compute_dask_array(ingestor: Ingestor) -> None:
    ingestor.array.compute()


def _make_root_datamodule(paths: List[str], columns: List[str]) -> Any:
    """Build a `ROOTPaddedDataModule` over the same files and columns as the other strategies.

    `ROOTPaddedDataModule` requires a non-empty `labels_columns`, which the other four strategies
    have no equivalent for. Reusing the last feature column as the target keeps the number of
    *distinct* columns read identical to the other strategies (`len(columns)`); RDataLoader reads
    it once per event loop and copies it into the target tensor.
    """
    from needle.ml.lightning.datamodules.root_padded_datamodule import ROOTPaddedDataModule

    return ROOTPaddedDataModule(
        dataset_config={
            "paths": paths,
            "features_columns": columns,
            "labels_columns": [columns[-1]],
            "format": "root",
            "max_number_events": -1,
        },
        tree_name=TREE_NAME,
        batch_size=RDATALOADER_BATCH_SIZE,
        batches_in_memory=RDATALOADER_BATCHES_IN_MEMORY,
        max_vec_sizes={column: MAX_VEC_SIZES[column.split(".")[0]] for column in columns},
        vec_padding=0.0,
        test_size=0,
        # The other strategies read sequentially; shuffling here would be extra work for one side.
        shuffle=False,
    )


@pytest.fixture()
def root_paths(delphes_sample_root: str) -> List[str]:
    return resolve_paths(delphes_sample_root)


@pytest.fixture()
def parquet_paths(delphes_sample_parquet: str) -> List[str]:
    return resolve_paths(delphes_sample_parquet)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_root_dask_setup(benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str) -> None:
    """Upfront cost of `Ingestor(format="root")`: `uproot.dask` graph/form construction across
    every input file, plus `eager_compute_divisions()`. No event data is read.
    """
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]

    def _setup() -> Ingestor:
        return Ingestor(
            paths=paths, format="root", columns=columns, reader_kwargs={"filter_name": _filter_name(columns)}
        )

    benchmark.pedantic(_setup, rounds=NUM_ROUNDS, iterations=NUM_ITERATIONS, warmup_rounds=NUM_WARMUP_ROUNDS)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_root_dask_read(benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str) -> None:
    """Cost of reading all requested columns from an already-built ROOT `Ingestor` -- a single
    `ingestor.array.compute()`, the one dask-graph execution a real consumer triggers.
    """
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]
    ingestor = Ingestor(
        paths=paths, format="root", columns=columns, reader_kwargs={"filter_name": _filter_name(columns)}
    )

    benchmark.pedantic(
        _compute_dask_array,
        args=(ingestor,),
        rounds=NUM_ROUNDS,
        iterations=NUM_ITERATIONS,
        warmup_rounds=NUM_WARMUP_ROUNDS,
    )


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_root_iterative_setup(
    benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Upfront cost of `IterableIngestor`: `uproot.num_entries` (TTree header only, all files) plus
    a single-file field lookup. No event data is read, and no dask graph is ever built.
    """
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]

    def _setup() -> IterableIngestor:
        return IterableIngestor(paths=paths, columns=columns, format="root")

    benchmark.pedantic(_setup, rounds=NUM_ROUNDS, iterations=NUM_ITERATIONS, warmup_rounds=NUM_WARMUP_ROUNDS)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_root_iterative_read(
    benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Cost of a full `uproot.iterate` pass over all files via `IterableIngestor.iterate()`."""
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]
    ingestor = IterableIngestor(paths=paths, columns=columns, format="root")

    def _read() -> None:
        for _ in ingestor.iterate():
            pass

    benchmark.pedantic(_read, rounds=NUM_ROUNDS, iterations=NUM_ITERATIONS, warmup_rounds=NUM_WARMUP_ROUNDS)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_parquet_dask_setup(
    benchmark: BenchmarkFixture, parquet_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Upfront cost of `Ingestor(format="parquet")`: `dak.from_parquet` graph construction (cheap
    `pyarrow.parquet.ParquetFile` metadata per file) plus `eager_compute_divisions()`.
    """
    columns = COLUMN_SETS[column_mode]
    paths = parquet_paths[:num_files]

    def _setup() -> Ingestor:
        return Ingestor(paths=paths, format="parquet", columns=columns)

    benchmark.pedantic(_setup, rounds=NUM_ROUNDS, iterations=NUM_ITERATIONS, warmup_rounds=NUM_WARMUP_ROUNDS)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_parquet_dask_read(
    benchmark: BenchmarkFixture, parquet_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Cost of reading all requested columns from an already-built parquet `Ingestor` -- a single
    `ingestor.array.compute()`, the one dask-graph execution a real consumer triggers.
    """
    columns = COLUMN_SETS[column_mode]
    paths = parquet_paths[:num_files]
    ingestor = Ingestor(paths=paths, format="parquet", columns=columns)

    benchmark.pedantic(
        _compute_dask_array,
        args=(ingestor,),
        rounds=NUM_ROUNDS,
        iterations=NUM_ITERATIONS,
        warmup_rounds=NUM_WARMUP_ROUNDS,
    )


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_parquet_iterative_setup(
    benchmark: BenchmarkFixture, parquet_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Upfront cost of `IterableIngestor`: `pyarrow.parquet.ParquetFile` metadata per file,
    no dask graph at all.
    """
    columns = COLUMN_SETS[column_mode]
    paths = parquet_paths[:num_files]

    def _setup() -> IterableIngestor:
        return IterableIngestor(paths=paths, columns=columns, format="parquet")

    benchmark.pedantic(_setup, rounds=NUM_ROUNDS, iterations=NUM_ITERATIONS, warmup_rounds=NUM_WARMUP_ROUNDS)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_parquet_iterative_read(
    benchmark: BenchmarkFixture, parquet_paths: List[str], num_files: int, column_mode: str
) -> None:
    """Cost of a full `ak.from_parquet`-per-file pass via `IterableIngestor.iterate()`."""
    columns = COLUMN_SETS[column_mode]
    paths = parquet_paths[:num_files]
    ingestor = IterableIngestor(paths=paths, columns=columns, format="parquet")

    def _read() -> None:
        for _ in ingestor.iterate():
            pass

    benchmark.pedantic(_read, rounds=NUM_ROUNDS, iterations=NUM_ITERATIONS, warmup_rounds=NUM_WARMUP_ROUNDS)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_rdataloader_setup(benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str) -> None:
    """Upfront cost of `ROOTPaddedDataModule.setup()`: `ROOT.RDataFrame` construction plus
    `RDataLoader` creation, which reads the TTree schema only. No event data is read.
    """
    pytest.importorskip("ROOT", reason="PyROOT not available; skipping the RDataLoader strategy")
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]

    def _setup() -> Any:
        datamodule = _make_root_datamodule(paths, columns)
        datamodule.setup()
        return datamodule

    benchmark.pedantic(_setup, rounds=NUM_ROUNDS, iterations=NUM_ITERATIONS, warmup_rounds=NUM_WARMUP_ROUNDS)


@pytest.mark.parametrize("column_mode", ["few", "many"])
@pytest.mark.parametrize("num_files", NUM_FILES)
def test_rdataloader_read(benchmark: BenchmarkFixture, root_paths: List[str], num_files: int, column_mode: str) -> None:
    """Cost of a full pass over every batch of an already-built `ROOTPaddedDataModule`, i.e. every
    requested column read and converted to padded torch tensors.
    """
    pytest.importorskip("ROOT", reason="PyROOT not available; skipping the RDataLoader strategy")
    columns = COLUMN_SETS[column_mode]
    paths = root_paths[:num_files]
    datamodule = _make_root_datamodule(paths, columns)
    datamodule.setup()

    def _read() -> None:
        for _ in datamodule.train_dataloader():
            pass

    benchmark.pedantic(_read, rounds=NUM_ROUNDS, iterations=NUM_ITERATIONS, warmup_rounds=NUM_WARMUP_ROUNDS)
