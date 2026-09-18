"""
Compare the speed up between root and parquet files on the same Delphes dataset. Requires that
the Environment variable for both the root and parquet version of the datasets are defined. Will
convert the root files to parquet if not already done so.

Disclaimer: Part of this code was written with the help of GPT-5 and Claude

Run these tests using the following command:

```python3
pytest tests/benchmarks/test_root_vs_parquet.py --benchmark-only -s -m "not slow"
pytest tests/benchmarks/test_root_vs_parquet.py --benchmark-only -s -m slow
```

`tests/benchmarks/conftest.py` derives the `--benchmark-json` output path from the `-m` mark
expression above (`results/root_vs_parquet_fast.json` / `_slow.json`) and creates
`tests/benchmarks/results/` if needed. No need to pass `--benchmark-json` explicitly.

The pytest mark `benchmark` is automatically added with the pytest fixture of the same name. This
test suite requires the specific Delphes dataset from KIT. There are two environment variables to
set:

```
export DELPHES_DATA_ROOT=/path/to/*.root
export DELPHES_DATA_PARQUET=/path/to/*.parquet
```

These must be a glob pattern of all the files. The columns and other configs are read from the
dedicated test `conf_tests` directory for all tests. In that config it is not mandatory to set
the paths to the datasets because they are overwritten by the two environment variables mentioned
above.
"""

import os
import re
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Literal, cast

import pydantic
import pytest
from omegaconf import OmegaConf
from pytest_benchmark.fixture import BenchmarkFixture
from torch.utils.data import DataLoader

from needle.etl.array import resolve_paths
from needle.etl.conversion import convert_root_to_parquet
from needle.etl.dask_ingestor import Ingestor
from needle.ml.datasets import PaddedDaskDataset
from needle.utils.config_schema import DatasetConfig, EstimatorConfig, ExpansionConfig

Percentage = Annotated[float, pydantic.Field(ge=0.0, le=100.0)]


DELPHES_DATASET_CONFIG = Path(__file__).parents[1] / "conf_tests" / "datasets" / "delphes.yaml"

FileType = Literal["parquet", "root", "rdataloader"]


BENCH_BATCH_SIZE = int(os.getenv("BENCH_BATCH_SIZE", "1"))
RDATALOADER_BATCHES_IN_MEMORY = int(
    os.getenv("RDATALOADER_BATCHES_IN_MEMORY", str(max(2, 1024 // BENCH_BATCH_SIZE)))
)
# Delphes collections are jagged, and RDataLoader materialises them at a fixed width. These are the
# global maxima of the `<Collection>_size` counter branches over the dataset; anything smaller
# silently truncates events.
MAX_VEC_SIZES: Dict[str, int] = {"Track": 253, "Photon": 3}
DELPHES_TREE_NAME = os.getenv("DELPHES_TREE_NAME", "Delphes")

def collection_of(column: str) -> str:
    """`Track.PT` and `Track_PT` both belong to the `Track` collection."""
    return re.split(r"[._]", column, maxsplit=1)[0]


@pytest.fixture()
def benchmark_config() -> EstimatorConfig:
    """Fixture that provides a standalone EstimatorConfig instance for benchmark tests.

    The dataset entry (most importantly the feature columns) is read from the dedicated test
    config directory `DELPHES_DATASET_CONFIG`, so that the benchmarks use the same columns as the
    rest of the test suite.

    Returns:
        EstimatorConfig
    """
    dataset_config = cast(
        DatasetConfig,
        OmegaConf.to_object(
            OmegaConf.merge(
                OmegaConf.structured(DatasetConfig),
                OmegaConf.load(DELPHES_DATASET_CONFIG),
            )
        ),
    )
    # `paths` and `max_number_events` are overridden by the test itself
    dataset_config.paths = ""
    dataset_config.max_number_events = -1

    estimator_config = EstimatorConfig(
        datamodule="default",
        datamodule_override=None,
        dataset="default",
        dataset_override=dataset_config,
        model="default",
        model_override=None,
        trainer="default",
        trainer_override=None,
        expands=ExpansionConfig(),
        requires=None,
    )

    return estimator_config


class BenchmarkUtility:
    COLUMN_MODES = {
        pytest.param("one", id="columns_1"),
        pytest.param("config", id="columns_config"),
    }
    FILE_PERCENTAGE = [
        pytest.param(0.0, id="files_0percent"),
        pytest.param(10.0, id="files_10percent", marks=pytest.mark.slow),
        pytest.param(50.0, id="files_50percent", marks=pytest.mark.slow),
        pytest.param(100.0, id="files_100percent", marks=pytest.mark.slow),
    ]
    NUM_EVENTS = [
        pytest.param(10**3, id="events_1k"),
        pytest.param(10**5, id="events_100k", marks=pytest.mark.slow),
        pytest.param(10**7, id="events_10M", marks=pytest.mark.slow),
        pytest.param(-1, id="events_all", marks=pytest.mark.slow),
    ]

    @staticmethod
    def get_column(
        column_mode: str,
        columns: List[str] | None,
        drop_branches: List[str] = ["fBits"],
    ) -> List[str] | None:
        """BUG `drop_branches` will not be applied if `columns == None`"""
        if columns:
            columns = [col for col in columns if all(drop not in col for drop in drop_branches)]
        match column_mode:
            case "one":
                return [columns[0]] if columns else None
            case "config":
                return columns
            case "all" | None:
                return None

    @staticmethod
    @pydantic.validate_call
    def get_files(file_percentage: Percentage, paths: List[str]) -> List[str]:
        """`file_percentage` is a percentage (0-100), not a fraction (0-1)."""
        return paths[: max(1, int(len(paths) * file_percentage / 100))]


def run_test(
    method: Literal["only_metadata", "materialize_partitions", "iterate_dataloader"],
    config: EstimatorConfig,
    paths: List[str],
    drop_branches: List[str],
    file_type: FileType,
) -> Callable:
    """Benchmark between root and parquet file ingestion with dask_awkward

    Args:
        method (Literal[&quot;only_metadata&quot;, &quot;materialize_partitions&quot;, &quot;iterate_dataloader&quot;]):
            Which kind of test to run from the list of implemented functions.
        config (MainConfig):
        paths (List[str]): List of paths to the data files. Valid paths are all paths accepted by `Ingestor`
        drop_branches (List[str]): List of potentially corrupted branches to drop at runtime

    Returns:
        Callable: A function without args that will run the desired test
    """

    assert config.dataset_override is not None

    def filter_name_func(columns: List[str]) -> Callable[[str], bool]:
        """Check if the str is in the list of branches to drop"""

        def _filter(name: str) -> bool:
            is_valid = not any(d in name for d in drop_branches)
            is_in_columns = name in columns
            return is_valid and is_in_columns

        return _filter

    def split_columns() -> tuple:
        """Split the configured feature columns into (features, labels).

        The config's own `labels_columns` (Photon.*) cannot be used as the label side here:
        `PaddedDatasetBase` caches a single `_feature_padding_length` and applies it to features
        and labels alike, and `add_innermost_dimension` takes a different branch for collections
        with empty events (Photon is empty in most events, Track never is). Mixing the two
        collections therefore dies in `ak.to_numpy` with "subarray lengths are not regular".
        That is very likely why upstream passes the *same* ingestor as both features and labels.

        Holding the last configured column out as the target keeps both sides within one
        collection, so every backend reads exactly `len(columns)` distinct columns once, and both
        emit the same (x, y) split. With a single configured column there is nothing to hold out,
        so upstream's duplicate-ingestor behaviour is kept for that parametrization.
        """
        assert config.dataset_override is not None
        columns = list(config.dataset_override.features_columns or [])
        if len(columns) < 2:
            return columns, columns
        return columns[:-1], columns[-1:]

    def reader_kwargs() -> Dict[str, Callable]:
        assert config.dataset_override is not None
        assert config.dataset_override.features_columns is not None
        match file_type:
            case "root":
                return {"filter_name": filter_name_func(config.dataset_override.features_columns)}
            case _:
                # parquet needs none; the rdataloader arm bypasses the Ingestor entirely
                return {}

    def _test_only_metadata():
        """Test function to read the metadata from the files

        Does not materialize partitions and does not perform any computation of the arrays.
        """
        assert config.dataset_override is not None
        _ = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.dataset_override.features_columns,
            max_number_events=config.dataset_override.max_number_events,
            reader_kwargs=reader_kwargs(),
        )

    def _test_materialize_partitions():
        """Test function to materialize partitions from ingested data.

        Creates an Ingestor instance with specified configuration, filters columns
        based on a filter function, and computes the mapped partitions to materialize
        them in memory. Performs no actual calculation.
        """
        assert config.dataset_override is not None
        ingestor = Ingestor(
            paths=paths,
            format="automatic",
            columns=config.dataset_override.features_columns,
            max_number_events=config.dataset_override.max_number_events,
            reader_kwargs=reader_kwargs(),
        )
        for field in ingestor.fields:
            ingestor[field].compute()

    def _test_iterate_dataloader():
        """Test function to iterate through a dataloader with padded dataset.

        This function creates an Ingestor instance and filters the columns based on the filter
        function, combines them into a PaddedDaskDataset, and then iterates through a DataLoader to
        verify that the data pipeline works correctly without errors.

        The test verifies that:
        - Data can be loaded and filtered properly
        - The DataLoader can iterate through the dataset without exceptions
        """
        assert config.dataset_override is not None
        features, labels = split_columns()
        features_ingestor = Ingestor(
            paths=paths,
            format="automatic",
            columns=features,
            max_number_events=config.dataset_override.max_number_events,
            reader_kwargs=({"filter_name": filter_name_func(features)} if file_type == "root" else {}),
        )
        # Upstream passed the *same* ingestor as both features and labels, which makes the dask
        # arms compute and tensor-convert the identical columns twice per partition while the
        # rdataloader arm reads features + targets once each. Using the config's real
        # `labels_columns` keeps every backend reading the same column set exactly once.
        labels_ingestor = Ingestor(
            paths=paths,
            format="automatic",
            columns=labels,
            max_number_events=config.dataset_override.max_number_events,
            reader_kwargs=({"filter_name": filter_name_func(labels)} if file_type == "root" else {}),
        )
        datamodule = PaddedDaskDataset(features_ingestor, labels_ingestor)
        dataloader = DataLoader(datamodule, batch_size=BENCH_BATCH_SIZE)

        for _ in dataloader:
            pass

    # --- RDataLoader equivalents of the three stages above ------------------
    # Same three depths, same .root files, but read through ROOT's native RDataLoader
    # (wrapped in ROOTPaddedDataModule) instead of uproot -> dask_awkward -> PaddedDaskDataset.

    def _make_root_datamodule() -> Any:
        """Build a ROOTPaddedDataModule over the same paths/columns as the dask arms."""
        from needle.ml.lightning.datamodules.root_padded_datamodule import ROOTPaddedDataModule

        assert config.dataset_override is not None
        features = list(config.dataset_override.features_columns or [])
        _, labels = split_columns()

        return ROOTPaddedDataModule(
            dataset_config={
                "paths": paths,
                "features_columns": features,
                "labels_columns": labels,
                "format": "root",
                "max_number_events": config.dataset_override.max_number_events,
            },
            tree_name=DELPHES_TREE_NAME,
            batch_size=BENCH_BATCH_SIZE,
            batches_in_memory=RDATALOADER_BATCHES_IN_MEMORY,
            max_vec_sizes={c: MAX_VEC_SIZES[collection_of(c)] for c in features + labels},
            vec_padding=0.0,
            test_size=0,
            # PaddedDaskDataset reads sequentially here, so shuffling would be extra work for one side.
            shuffle=False,
            # Emit (B, P, F) like PaddedDaskDataset, so both backends hand a model the same thing.
            particle_major=True,
        )

    def _test_rdataloader_only_metadata():
        """Counterpart of `_test_only_metadata`: open the files and build the lazy reader.

        `ROOT.RDataFrame` + `RDataLoader` construction only reads the tree schema; no event data
        is decompressed, matching what dask's graph building does.
        """
        _make_root_datamodule().setup()

    def _test_rdataloader_materialize():
        """Counterpart of `_test_materialize_partitions`: pull whole columns into memory.

        RDataLoader has no partition-materialization step (it streams fixed-size windows), so the
        closest analogue is `RDataFrame.AsNumpy()`, which reads whole columns eagerly with no
        batching and no tensor conversion -- the role `ingestor[field].compute()` plays for dask.
        One AsNumpy call per column, mirroring the dask arm's one-compute-per-field loop, each over
        an RDataFrame with only that branch enabled (see `make_rdataframe`).
        """
        from needle.ml.lightning.datamodules.root_padded_datamodule import make_rdataframe

        assert config.dataset_override is not None
        for column in list(config.dataset_override.features_columns or []):
            rdf, _chain = make_rdataframe(DELPHES_TREE_NAME, paths, [column])
            if config.dataset_override.max_number_events > 0:
                rdf = rdf.Filter(f"rdfentry_ < {config.dataset_override.max_number_events}")
            _ = rdf.AsNumpy([column])

    def _test_rdataloader_iterate():
        """Counterpart of `_test_iterate_dataloader`: full pass over every batch.

        This is the only stage that is a genuine like-for-like comparison: both sides hand back
        padded float32 tensors ready for a model.
        """
        datamodule = _make_root_datamodule()
        datamodule.setup()

        for _ in datamodule.train_dataloader():
            pass

    test_methods = {
        "only_metadata": _test_only_metadata,
        "materialize_partitions": _test_materialize_partitions,
        "iterate_dataloader": _test_iterate_dataloader,
    }
    rdataloader_methods = {
        "only_metadata": _test_rdataloader_only_metadata,
        "materialize_partitions": _test_rdataloader_materialize,
        "iterate_dataloader": _test_rdataloader_iterate,
    }

    if file_type == "rdataloader":
        return rdataloader_methods[method]

    return test_methods[method]


@pytest.mark.parametrize("file_percentage", BenchmarkUtility.FILE_PERCENTAGE)
@pytest.mark.parametrize("column_mode", BenchmarkUtility.COLUMN_MODES)
@pytest.mark.parametrize("num_events", BenchmarkUtility.NUM_EVENTS)
@pytest.mark.parametrize("file_type", ["root", "parquet", "rdataloader"])
@pytest.mark.parametrize("test_method", ["only_metadata", "materialize_partitions", "iterate_dataloader"])
def test_ingestion_speed(
    benchmark: BenchmarkFixture,
    benchmark_config: EstimatorConfig,
    delphes_sample_root: str,
    delphes_sample_parquet: str,
    column_mode: str,
    file_percentage: Percentage,
    file_type: FileType,
    test_method: Literal["only_metadata", "materialize_partitions", "iterate_dataloader"],
    num_events: int,
    drop_branches=["ref", "fName", "fSize", "fP", "fE", "fBits"],
) -> None:
    """Test function to compare the ingestion speeds of parquet and root files in different scenarios.

    The larger and longer test for many events are marked as slow. To run them, add the '-m slow' marker
    when running the tests. More info is given in the corresponding `BenchmarkUtility` class.

    Args:
        benchmark (BenchmarkFixture): Registers this test as a pytest-benchmark instance
        benchmark_config (EstimatorConfig): Configuration instance for benchmark tests
        delphes_sample_root (str): Path to the Delphes (Root) samples
        delphes_sample_parquet (str): Path to the Delphes (Parquet) samples. if empty, these files
            will be generated by converting the root files from `delphes_sample_root` to .parquet.
        column_mode (str): Which columns to choose. See `BenchmarkUtility`
        file_percentage (Percentage): How many files to open. See `BenchmarkUtility`
        file_type (str): Run this test either as Root or Parquet files.
        test_method (str): Which benchmark test method to run. See `BenchmarkUtility`
        num_events (int): How many events to load. Will cap at the maximal amount of events found in
            the loaded files.
        drop_branches (list, optional): Remove these branches when reading and converting files.
            Defaults to ["ref", "fName", "fSize", "fP", "fE", "fBits"], which are invalid branches
            in the default Delphes dataset.
    """
    if file_type == "rdataloader":
        pytest.importorskip("ROOT", reason="PyROOT not available; skipping the RDataLoader arm")

    if file_type == "parquet":
        convert_root_to_parquet(
            delphes_sample_root,
            Path(delphes_sample_parquet).parent,
            drop_branches=drop_branches,
        )
        data_path = delphes_sample_parquet
    else:
        # both "root" (uproot) and "rdataloader" (native ROOT) read the same .root files
        data_path = delphes_sample_root

    config: EstimatorConfig = benchmark_config
    # Access dataset config through the default estimator
    dataset_config = config.dataset_override
    assert dataset_config is not None
    dataset_config.max_number_events = num_events
    dataset_config.features_columns = BenchmarkUtility.get_column(
        column_mode=column_mode,
        columns=dataset_config.features_columns,
        drop_branches=drop_branches,
    )
    paths = BenchmarkUtility.get_files(
        file_percentage=file_percentage,
        paths=resolve_paths(data_path),
    )
    benchmark(
        run_test(
            method=test_method,
            config=config,
            paths=paths,
            drop_branches=drop_branches,
            file_type=file_type,
        ),
    )
