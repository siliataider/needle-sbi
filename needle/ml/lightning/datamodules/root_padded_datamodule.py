from typing import Any

import lightning as L

from needle.utils.config_schema import DatasetConfig


def _add_particle_dim(batch: Any) -> Any:
    x, y = batch
    return x.unsqueeze(1), y.unsqueeze(1)


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

    def _make_loader(self) -> Any:
        import ROOT
        from ROOT.Experimental.ML import RDataLoader

        rdf: Any = ROOT.RDataFrame(self.tree_name, self.dataset_config.paths)

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

    def train_dataloader(self):
        print("Creating train dataloader...")
        return map(_add_particle_dim, self._train.as_torch())

    def val_dataloader(self):
        return map(_add_particle_dim, self._val.as_torch())
