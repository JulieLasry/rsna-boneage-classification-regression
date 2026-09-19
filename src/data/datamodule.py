import os
import pandas as pd
from typing import Optional
import lightning as L
from torch.utils.data import DataLoader

from src.data.dataset import BoneAgeDataset

class DataModule(L.LightningDataModule):
    """
    Datamodule class implementing the DataLoader to build the
    list of Dataset sample identifiers: the PNG image path, the
    associated boneage and the gender category, if it's a male or
    not. The specific train, val and test dataloaders are created 
    with a given batch size.

    Args:
        fold_index (int): The current fold index.
        preprocessing_config (dict): Base config dictionnary defining the 
            main paths, tasks and parameter values for the preprocessing task.
        training_config (dict): Config dictionnary defining the 
            values of the training parameters (dataloader, train loop, etc.).
        raw_images_path (str): Path to the raw PNG files.
        csv (pd.DataFrame): Processed CSV with the train-
            val-test.
        
    """
    def __init__(
        self,
        fold_index: int,
        preprocessing_config: dict,
        training_config: dict,
        raw_images_path: str,
        csv: pd.DataFrame
    ) -> None:
        super().__init__()
        self.preprocessing_config = preprocessing_config
        self.training_config = training_config
        self.raw_images_path = raw_images_path
        self.csv = csv

        self.train_samples: list[tuple[str, int, int, bool]] = []
        self.val_samples: list[tuple[str, int, int, bool]] = []
        self.test_samples: list[tuple[str, int, int, bool]] = []

        self.test_dataset: Optional[BoneAgeDataset] = None
        self.train_dataset: Optional[BoneAgeDataset] = None
        self.val_dataset: Optional[BoneAgeDataset] = None

        for case in csv.index:
            png_path: str = os.path.join(raw_images_path, csv.loc[case, 'filename'])
            label: int = csv.loc[case, 'label']
            boneage: int = csv.loc[case, 'boneage']
            male: bool = csv.loc[case, 'male']
            is_test: bool = csv.loc[case, 'is_test']
            fold: int = csv.loc[case, 'fold']

            samples: tuple[str, int, int, bool] = png_path, label, boneage, male

            if is_test:
                self.test_samples.append(samples)
            else:
                if fold==fold_index:
                    self.val_samples.append(samples)
                else:
                    self.train_samples.append(samples)
             
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = BoneAgeDataset(
            samples=self.train_samples,
            preprocessing_config=self.preprocessing_config,
            is_train=True)

        if self.val_samples:
            self.val_dataset = BoneAgeDataset(
                samples=self.val_samples,
                preprocessing_config=self.preprocessing_config,
                is_train=False)
        else:
            self.val_dataset = None

        self.test_dataset = BoneAgeDataset(
            samples=self.test_samples,
            preprocessing_config=self.preprocessing_config,
            is_train=False)


    def train_dataloader(self) -> DataLoader:
        assert self.train_dataset is not None, \
            "train_dataset is None. Call setup()"
        return DataLoader(
            self.train_dataset,
            batch_size=self.training_config['dataloader']['batch_size'],
            num_workers=self.training_config['dataloader']['num_workers'],
            shuffle=True
        )

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(
            self.val_dataset,
            batch_size=self.training_config['dataloader']['batch_size'],
            num_workers=self.training_config['dataloader']['num_workers'],
            shuffle=False
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_dataset is not None, \
            "test_dataset is None. Call setup()"
        return DataLoader(
            self.test_dataset,
            batch_size=self.training_config['dataloader']['batch_size'],
            num_workers=self.training_config['dataloader']['num_workers'],
            shuffle=False
        )
