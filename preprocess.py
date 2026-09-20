import yaml
from pathlib import Path
from src.data.raw_data_loading import RawDataLoading
from src.data.datamodule import DataModule

CONFIG_DIR: Path = Path(__file__).parent / 'configs'

class Preprocess():
    """
    Processer main function to load our raw data,
    process and transform this data, create dataloaders
    associated and print their length. Different steps 
    are defined for that.         
    """

    @staticmethod
    def load_config(path: Path) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def run() -> dict:
        # Configs paths loaded
        preprocesser_config: dict = Preprocess.load_config(
            CONFIG_DIR / 'preprocessing_config.yaml'
        )
        training_config: dict = Preprocess.load_config(
            CONFIG_DIR / 'training_config.yaml'
        )

        raw_images_path, csv = RawDataLoading.load_raw_data(preprocesser_config)
        RawDataLoading.split_raw_data(preprocesser_config, csv)

        dataloaders_dict: dict = {}

        # Creating and printing the length of the dataloaders
        for i in range(1, preprocesser_config['split']['n_folds']+1):
            dataloader = DataModule(
                fold_index=i,
                preprocessing_config=preprocesser_config,
                training_config=training_config,
                raw_images_path=raw_images_path,
                csv=csv
            )
            dataloader.setup()
            dataloaders_dict[i] = dataloader
            for split in ("train", "val", "test"):
                dataset = getattr(dataloader, f"{split}_dataset")
                if dataset is not None:
                    print(f"Fold {i} - {split} - dataset length : {len(dataset)}")

        return dataloaders_dict

if __name__ == '__main__':
    dataloaders_dict: dict = Preprocess.run()
