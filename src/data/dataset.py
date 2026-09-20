from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch

from src.data.transforms import Transform

class BoneAgeDataset(Dataset):
    """
    Resize all the PNG files to a target same size for
    processing same size inputs in all the batches. The
    target size represents a choosen 224x224 size, while 
    still preseving the image ratio. An interpolation method
    is also applied to keep this aspect. 
    It likewise manages the data with the number of channels
    and the type of data conversion. 
    It then returns four sample values: the PNG processed file, label,
    the boneage and the gender category.

    Args:
        samples (list[tuple[str, int, int, bool]): path of the png file,
            its label, boneage and if it's a male patient for the gender category.
        preprocessing_config (dict): Base config dictionnary defining the 
            main paths, tasks and parameter values for the preprocessing task.
        is_train (bool): If the loaded image is in the training set,
            for applying specific augmentations.
    """
    
    def __init__(
        self,
        samples: list[tuple[str, int, int, bool]],
        preprocessing_config: dict,
        is_train: bool = False,
    ) -> None:
        self.samples = samples
        self.is_train = is_train
        self.preprocessing_config = preprocessing_config

        # Transformations
        self.base_transform = Transform.base_transform(preprocessing_config=preprocessing_config)
        self.train_transform = Transform.train_transform(preprocessing_config=preprocessing_config)
        self.final_transform = Transform.final_transform(preprocessing_config=preprocessing_config)
        
    def __len__(
        self,
    ) -> int:
        return len(self.samples)

    def __getitem__(
        self, 
        idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Before loading the PNG file and their informations,
        label, boneage and gender category, several transformations
        are applied. We resize the image, convert to RGB
        and perform geometric and luminosity augmentations
        on the train set only. 

        Args:
            idx (int): The selected PNG file and information
            index.
            
        Returns:
            The tuple of the tensor PNG file, the label, boneage value
            and the gender category.
        """
        png_path, label, boneage, male = self.samples[idx]
        with Image.open(png_path) as png_file:
            # Resizing all images and adjusting gray-scale
            processed_file: np.ndarray = self.base_transform(image=np.asarray(
                png_file))['image']
            
            if self.is_train:
                # Apply horizontal flip, rotation and zoom
                processed_train_file: np.ndarray = self.train_transform(image=processed_file)['image']
                tensor_image: torch.Tensor = self.final_transform(image=processed_train_file)['image'] 

            else:
                tensor_image: torch.Tensor = self.final_transform(image=processed_file)['image']

        tensor_label: torch.Tensor = torch.tensor(label, dtype=torch.long)
        tensor_boneage: torch.Tensor = torch.tensor(boneage, dtype=torch.float32) 
        tensor_male: torch.Tensor = torch.tensor(male, dtype=torch.float32)
        
        return tensor_image, tensor_label, tensor_boneage, tensor_male
