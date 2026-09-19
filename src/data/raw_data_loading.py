import pandas as pd
import os 
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

class RawDataLoading():
    """
    Load the raw PNG files and the CSV data relative to the boneage.  
    Split also the IDs into train/val/test splits based on a 
    K-Fold cross validation.
    """
    
    @staticmethod
    def load_raw_data(
        preprocessing_config: dict,
    ) -> tuple[str, pd.DataFrame]:
        """
        Load the raw images and the csv data lines that have an ID (to
        correctly associate the image, gender category and boneage).  
        We also add a column to the CSV for the filename (with the .png 
        extension) and the boneage category.

        Args:
            preprocessing_config (dict): Base config dictionnary defining the 
                main paths, tasks and parameter values for the preprocessing task.

        Returns:
            The raw images path and CSV, having the gender category, 
            boneage and filename.
        """

        # Loading the raw images and CSV
        raw_data: pd.DataFrame = pd.read_csv(
            os.path.join(preprocessing_config['data']['base_dir'], 'boneage-training-dataset.csv'), 
            header=0)
        raw_images_path: str = os.path.join(
            preprocessing_config['data']['base_dir'], 
            'boneage-training-dataset/boneage-training-dataset')

        data: pd.DataFrame = raw_data.dropna().copy()

        # Selecting only the files having an ID (and therefore a gender and boneage)
        available: set = set(os.listdir(raw_images_path))
        data['filename'] = data['id'].astype(str) + '.png'
        csv: pd.DataFrame = data[data['filename'].isin(available)].reset_index(drop=True)

        # Categories : 0-5y (0) - 5-10y (1) - 10-15y (2) - 15-19y (3)
        labels: list[int] = [0, 1, 2, 3]
        thresholds: list[int] = [0, 60, 120, 180, np.inf]
        csv['label'] = pd.cut(csv['boneage'], thresholds, labels=labels)
        assert csv['label'].notna().all()

        return raw_images_path, csv 

    
    @staticmethod
    def split_raw_data(
        preprocessing_config: dict,
        csv: pd.DataFrame,
    ) -> None :
        """
        Split into train/val/test categories the different 
        files of the loaded raw CSV. This CSV must contain 
        the new added column in the load_raw_data() function 
        above. A K-Fold  cross-validation is also implemented 
        here, by allocating each train/val file a fold, to 
        better estimate our model's robustness.

        Args:
            preprocessing_config (dict): Base config dictionnary defining the 
                main paths, tasks and parameter values for the preprocessing task.
            csv (pd.DataFrame): The raw CSV with new columns. 

        Returns:
            None.
        """

        # Split the data into train and test csv 
        csv_train_raw, csv_test = train_test_split(
            csv,
            test_size=preprocessing_config['split']['test_size'],
            random_state=preprocessing_config['split']['seed'],
            stratify=csv['label']
        )

        # Obtain the K(5)-Fold CV indices
        csv_train: pd.DataFrame = csv_train_raw.reset_index(drop=True)
        cv_indices = StratifiedKFold(
            n_splits=preprocessing_config['split']['n_folds'], 
            random_state=preprocessing_config['split']['seed'], 
            shuffle=True
        )

        # Create Test and Fold colomns in the global CSV
        csv['fold'] = -1
        for fold_num, (_, val_idx) in enumerate(
            cv_indices.split(csv_train, csv_train['label']), 1
        ):
            val_ids: pd.Series = csv_train.iloc[val_idx]['id']
            csv.loc[csv['id'].isin(val_ids), 'fold'] = fold_num

        csv['is_test'] = False
        csv.loc[csv['id'].isin(csv_test['id']), 'is_test'] = True
        
        # Saving the results
        csv_output_file: str = os.path.join(preprocessing_config['data']['output_dir'], 'data.csv')
        os.makedirs(preprocessing_config['data']['output_dir'], exist_ok=True)
        csv.to_csv(csv_output_file, index=False)
