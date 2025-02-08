# Machine Learning Final Project

This repository contains the implementation of a comparative study between our model and baseline CAT method, focusing on different hyperparameters and model architectures.

## Project Structure

```
.
├── code/                  # Source code
│   ├── Model.py          # Model implementations
│   ├── Preprocessing.py  # Data preprocessing utilities
│   ├── SSPMI_emb.py     # SSPMI embedding generation
│   └── run_experiments.py# Main experiment runner
├── results/              # Experiment results and visualizations
└── releases/            # Data and documentation files
```

## Features

- Multiple model architectures (Linear, CNN, Transformer)
- Different embedding types (Naive, SSPMI)
- Various hidden dimensions (64, 128, 256)
- Confidence-aware predictions
- Comprehensive experiment tracking and visualization

## Data Access

Due to file size limitations, the data files and documentation are available through GitHub Releases:

1. Download the required files from the [Releases page](https://github.com/1hansir/ML_CAT/releases)
   - `data.zip`: Contains all data files including training, validation, and inference results
   - `documents.zip`: Contains project documentation and related papers

2. Extract the files to their respective directories:
   ```bash
   # Extract data files
   unzip data.zip
   
   # Extract documentation
   unzip documents.zip
   ```

## Setup

1. Create a conda environment:
```bash
conda create -n ml_project python=3.8
conda activate ml_project
```

2. Install dependencies:
```bash
conda install pytorch torchvision torchaudio -c pytorch
conda install pandas numpy matplotlib seaborn scikit-learn tqdm
```

## Running Experiments

To run all experiments:
```bash
python code/run_experiments.py
```

This will:
1. Train models with different configurations
2. Run inference on test data
3. Generate visualizations and analysis
4. Save results in the results/ directory

## Results

The experiments compare:
1. Hidden vector dimensions (64, 128, 256)
2. Model architectures (Linear, CNN, Transformer)
3. Initial embeddings (Naive vs SSPMI)

Results are saved as:
- CSV files with metrics
- Visualization plots
- Model checkpoints
- Inference predictions 