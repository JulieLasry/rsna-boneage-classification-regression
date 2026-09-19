### Bone Age Classification / Regression

## Project scope : 
Predicting a child's bone age from hand radiographs, using the 
[RSNA Bone Age dataset](https://www.kaggle.com/datasets/kmader/rsna-bone-age) dataset, 
available on Kaggle. We have just selcted 2000 images on 12.8K files, to easily process
the data.
The task can be either a **classification** (age groups)
or a **regression** (age in months), based on one congif parameter value.
The reprository currently includes a complete data processing pipeline (
loading, stratified K-Fold cross validation (K=5), augmentations,
Lightning DataModules). The model training and evaluation are in progress.

## Installation

Requires Python 3.11.

**macOS / Linux**
```bash
git clone <https://github.com/JulieLasry/rsna-boneage-classification-regression>
cd Bone_Age_Classification_Regression
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows**
```powershell
git clone <https://github.com/JulieLasry/rsna-boneage-classification-regression>
cd Bone_Age_Classification_Regression
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Data.** Download the dataset from Kaggle and place it as follows:

```
data/raw/
├── boneage-training-dataset.csv
└── boneage-training-dataset/boneage-training-dataset/*.png
```

## Execution and usage

Run the data pipeline from the project root:

- For data processing : 
```bash
python processer.py
```
This creates `data/processed/data.csv` (fold and test assignment for each
image) and prints the size of the train / val / test sets for every fold.


Parameters are set in `configs/`:

| File | Content |
|------|---------|
| `preprocessing_config.yaml` | task (`classification` / `regression`), paths, image size, split, augmentations, normalization |
| `training_config.yaml` | dataloader and training parameters |

Project structure:

```
├── configs/          # YAML configs
├── data/             # raw and processed data (not versioned)
├── models/           # trained weights (not versioned)
├── notebooks/        # data exploration
├── src/data/         # raw loading, transforms, Dataset, DataModule
├── processer.py      # runs the data pipeline
└── train.py, test.py, inference.py, optimize.py
```

## Used technologies

| Technology | Explanation |
|---|---|
| PyTorch | Deep learning framework |
| Lightning | Structures the data and training code (`DataModule`, `LightningModule`) |
| Albumentations | Fast image augmentations and preprocessing |
| scikit-learn | Stratified splits and K-fold cross-validation |
| pandas / NumPy | Tabular data handling |
| Pillow | Image loading |
| PyYAML | Config files |

Planned: MLflow (experiment tracking), Optuna (hyperparameter search).

## Current features

- Config-driven pipeline, with no hard-coded hyperparameters
- Classification or regression target, chosen in the config
- Stratified test set, then stratified 5-fold cross-validation
- Train-only augmentations (flip, rotation, affine, noise, brightness/contrast)
- One Lightning `DataModule` per fold

## Contributing

This is a personal learning project, but suggestions are welcome:

1. Fork the repository and create a branch (`git checkout -b feature/my-idea`).
2. Commit your changes with a clear message.
3. Open a pull request describing the change.

## Contributors

<Julie Lasry> (author)

## Author

<Julie Lasry> — <juliexlasry@gmail.com>

## Change log

- **v0.1.0** — Initial data pipeline: raw loading, split, transforms,
  Dataset and DataModule.

