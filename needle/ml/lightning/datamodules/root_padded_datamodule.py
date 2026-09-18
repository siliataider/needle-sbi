from typing import Any

import lightning as L

from needle.utils.config_schema import DatasetConfig


def _add_particle_dim(batch: Any) -> Any:
    x, y = batch
    return x.unsqueeze(1), y.unsqueeze(1)


def _particle_major(batch: Any, max_vec_size: int, contiguous: bool = True) -> Any:
    """Reshape RDataLoader's flat per-event vector into `PaddedDataset`'s `(B, P, F)` layout.

    RDataLoader concatenates each padded column end to end, so an event arrives as
    `[col0 x P | col1 x P | ...]` (column-major, verified against the source branches). Splitting
    that into `(B, F, P)` and transposing the last two axes yields the same `(batch, particles,
    features)` layout `PaddedDatasetBase.convert_ak_to_tensor` produces, so both backends hand a
    model the identical thing.

    Only valid when every column shares one padding width -- with per-collection widths (e.g.
    Track=253, Photon=3) there is no single `P` and the flat layout is the only correct one.

    Args:
        batch: `(x, y)` as produced by `RDataLoader.as_torch()`, each `(B, n_columns * P)`.
        max_vec_size: The shared padding width `P`.
        contiguous: Materialise the transpose. `transpose` alone returns a non-contiguous view,
            which would hide the cost of the layout change from any benchmark timing this.

    Returns:
        tuple: `(x, y)` shaped `(B, P, F)`.
    """

    def _reshape(tensor: Any) -> Any:
        n_columns = tensor.shape[-1] // max_vec_size
        out = tensor.view(tensor.shape[0], n_columns, max_vec_size).transpose(1, 2)
        return out.contiguous() if contiguous else out

    x, y = batch
    return _reshape(x), _reshape(y)


def make_rdataframe(tree_name: str, paths: Any, columns: list[str]) -> tuple[Any, Any]:
    """Build an RDataFrame over `paths` that only reads `columns` from disk.

    `ROOT.RDataFrame(tree_name, paths)` leaves every branch enabled. For split collections such as
    Delphes' `TClonesArray`s, reading one member (`Track.PT`) then deserialises every enabled member
    of that collection (all 52 for `Track`). Disabling everything and
    re-enabling the requested branches restores member-wise reading, and is a no-op for trees whose
    columns are independent top-level branches.

    Returns:
        tuple: `(rdf, chain)`. The RDataFrame does not own the TChain, so keep the chain alive for as
        long as the RDataFrame is used.
    """
    import ROOT

    chain = ROOT.TChain(tree_name)
    for path in [paths] if isinstance(paths, str) else paths:
        chain.Add(str(path))
    chain.SetBranchStatus("*", 0)
    for column in columns:
        chain.SetBranchStatus(column, 1)
    return ROOT.RDataFrame(chain), chain


class ROOTPaddedDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_config: dict,
        tree_name: str,
        batch_size: int = 256,
        batches_in_memory: int = 10,
        max_vec_sizes: dict[str, int] | None = None,
        vec_padding: float = 0.0,
        test_size: float = 0.2,
        shuffle: bool = True,
        set_seed: int = 0,
        drop_remainder: bool = False,
        particle_major: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_config = DatasetConfig(**dataset_config)
        self.tree_name = tree_name
        self.batch_size = batch_size
        self.batches_in_memory = batches_in_memory
        self.max_vec_sizes = max_vec_sizes
        self.vec_padding = vec_padding
        self.test_size = test_size
        self.shuffle = shuffle
        self.set_seed = set_seed
        self.drop_remainder = drop_remainder
        # Emit (B, P, F) like PaddedDataset instead of RDataLoader's flat (B, 1, F*P). Requires a
        # single shared padding width; see `_particle_major`.
        self.particle_major = particle_major
        # TChains backing the RDataFrames built in `_make_loader`; see `make_rdataframe`.
        self._chains: list[Any] = []

    def _make_loader(self) -> Any:
        from ROOT.Experimental.ML import RDataLoader

        columns = list(dict.fromkeys(self.dataset_config.features_columns + self.dataset_config.labels_columns))
        rdf, chain = make_rdataframe(self.tree_name, self.dataset_config.paths, columns)
        self._chains.append(chain)

        if self.dataset_config.max_number_events > 0:
            rdf = rdf.Filter(f"rdfentry_ < {self.dataset_config.max_number_events}")

        return RDataLoader(
            rdf,
            batch_size=self.batch_size,
            batches_in_memory=self.batches_in_memory,
            columns=self.dataset_config.features_columns,
            max_vec_sizes=self.max_vec_sizes,
            vec_padding=self.vec_padding,
            target=self.dataset_config.labels_columns,
            shuffle=self.shuffle,
            set_seed=self.set_seed,
            drop_remainder=self.drop_remainder,
        )

    def setup(self, stage: str | None = None) -> None:
        print("Setting up ROOTPaddedDataModule...")

        assert self.dataset_config.features_columns
        assert self.dataset_config.labels_columns

        loader = self._make_loader()
        print("RDataLoader created.")

        if self.test_size > 0:
            self._train, self._val = loader.train_test_split(test_size=self.test_size)
        else:
            self._train = loader
            self._val = self._make_loader()

    def _shape_batches(self, loader: Any) -> Any:
        """Apply the configured output layout to every batch of `loader`."""
        if not self.particle_major:
            return map(_add_particle_dim, loader.as_torch())

        widths = set((self.max_vec_sizes or {}).values())
        if len(widths) != 1:
            raise ValueError(
                "particle_major=True needs one shared max_vec_size across all columns, got "
                f"{sorted(widths)}. With per-collection widths there is no single particle axis."
            )
        max_vec_size = widths.pop()
        return map(lambda batch: _particle_major(batch, max_vec_size), loader.as_torch())

    def train_dataloader(self):
        print("Creating train dataloader...")
        return self._shape_batches(self._train)

    def val_dataloader(self):
        return self._shape_batches(self._val)