# CPM: Conditioned Prior Modeling for Hyperspectral Image Classification

This repository contains the implementation of **Conditioned Prior Modeling (CPM)** for few-shot cross-domain hyperspectral image classification.

The codebase provides dataset-specific training entry points for **Indian Pines**, **Salinas**, **University of Pavia**, and **Houston**, together with the CPM modules for category-centric semantic aggregation (CCSA), category-relational alignment (CRA), label-conditioned perturbation (LCP), spectral-spatial feature extraction, prototype refinement, evaluation, and visualization.

## Project structure

```text
Stt_CPM/
├── cpm/
│   ├── ccsa/                    # Category-centric semantic aggregation
│   ├── cra/                     # Category-relational alignment
│   ├── lcp/                     # Label-conditioned perturbation
│   ├── data/                    # Data loading and preprocessing
│   ├── evaluation/              # Target-domain evaluation
│   ├── feature_extraction/      # Spectral-spatial feature extraction
│   ├── prototype_refinement/    # Prototype refinement network
│   ├── runtime/                 # Device, logging, initialization, reproducibility
│   ├── training/                # Episodic CPM training
│   └── visualization/           # Classification maps and t-SNE
├── configs/
│   ├── houston.py
│   ├── indian_pines.py
│   ├── salinas.py
│   └── university_of_pavia.py
├── datasets/
│   └── README.md
├── pretrained_models/
│   └── all-mpnet-base-v2/
│       └── README.md
├── train_cpm_houston.py
├── train_cpm_indian_pines.py
├── train_cpm_salinas.py
├── train_cpm_university_of_pavia.py
├── PROJECT_STRUCTURE.md
├── requirements.txt
└── README.md
```

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The current requirements are:

- NumPy
- SciPy
- scikit-learn
- h5py
- Matplotlib
- PyTorch
- TensorBoard
- Transformers
- PEFT

A CUDA-capable GPU is recommended for the full training procedure. The GPU index can be changed in the corresponding file under `configs/`.

## Dataset preparation

Dataset binaries are **not included** in this repository. Place the data under `datasets/` using the following structure:

```text
datasets/
├── chikusei/
│   └── chikusei_source_patches_128bands_7x7.pkl
├── houston/
│   ├── houston_hyperspectral_image.mat
│   └── houston_ground_truth.mat
├── indian_pines/
│   ├── indian_pines_hyperspectral_image.mat
│   └── indian_pines_ground_truth.mat
├── salinas/
│   ├── salinas_hyperspectral_image.mat
│   └── salinas_ground_truth.mat
└── university_of_pavia/
    ├── university_of_pavia_hyperspectral_image.mat
    └── university_of_pavia_ground_truth.mat
```

The MATLAB variable names expected by the code are defined explicitly in `configs/*.py`.

### Chikusei preprocessing

If you have the Chikusei image and ground-truth `.mat` files rather than the preprocessed `.pkl`, place them at:

```text
datasets/chikusei/chikusei_hyperspectral_image.mat
datasets/chikusei/chikusei_ground_truth.mat
```

Then run:

```bash
python -m cpm.data.chikusei_preprocessing
```

This creates:

```text
datasets/chikusei/chikusei_source_patches_128bands_7x7.pkl
```

## Pretrained MPNet model

CPM uses the Hugging Face `all-mpnet-base-v2` model. The model binaries are not included in this repository.

Place a complete local copy of the model in:

```text
pretrained_models/all-mpnet-base-v2/
```

Keep the standard Hugging Face filenames unchanged, for example `config.json`, tokenizer files, and model weights.

A complete local model directory is especially important when the training server cannot access `huggingface.co`; otherwise `transformers` may attempt an online lookup and fail.

You can also override the configured model directory at runtime:

```bash
python train_cpm_indian_pines.py --model_path /path/to/all-mpnet-base-v2
```

## Training

Run the desired dataset entry point from the repository root.

### Indian Pines

```bash
python train_cpm_indian_pines.py
```

### Salinas

```bash
python train_cpm_salinas.py
```

### University of Pavia

```bash
python train_cpm_university_of_pavia.py
```

### Houston

```bash
python train_cpm_houston.py
```

Each entry point loads its default configuration from `configs/`.

## Useful command-line options

The training entry points support the following optional overrides:

```text
--config PATH          Override the dataset configuration file
--model_path PATH      Override the local pretrained MPNet directory
--num_runs N           Override the number of repeated experiment runs
--eval_interval N      Override the evaluation interval
--log_interval N       Override the logging interval
--enable_tsne          Enable t-SNE visualization during evaluation
```

For example, a single-run check on Indian Pines can be launched with:

```bash
python train_cpm_indian_pines.py --num_runs 1
```

To enable t-SNE:

```bash
python train_cpm_indian_pines.py --enable_tsne
```

## Default experimental settings

The four dataset configurations use 5 labeled target samples per class and 10 repeated runs by default. Dataset-specific CPM hyperparameters are stored directly in the corresponding configuration files, including:

- prompt virtual tokens
- LCP noise standard deviation
- CRA loss weight
- CRA temperature
- spectral dimensions
- class names and visualization colors
- random seeds

Refer to `configs/*.py` for the exact settings used by each dataset.

## Outputs

During training, generated artifacts are written to:

```text
logs/                         # Experiment logs
outputs/classification_maps/  # Classification maps
visualizations/tsne/          # t-SNE figures when enabled
```

These generated files are excluded from version control by `.gitignore`.

## Reproducibility

Each dataset configuration contains a fixed list of random seeds. By default, the code performs 10 experiment runs and reports aggregate evaluation statistics across those runs.

Before a formal experiment, verify that:

1. all required dataset files are present under `datasets/`;
2. the local MPNet directory contains the complete Hugging Face model assets;
3. the selected `gpu_id` is valid for your machine;
4. the Python dependencies in `requirements.txt` are installed.

## Quick syntax check

You can verify the Python source tree without dataset/model binaries using:

```bash
python -m compileall cpm configs \
  train_cpm_houston.py \
  train_cpm_indian_pines.py \
  train_cpm_salinas.py \
  train_cpm_university_of_pavia.py
```

## Notes

- Dataset binaries and pretrained model weights are intentionally excluded from this repository.
- Do not commit private paths, credentials, access tokens, generated logs, checkpoints, or experiment outputs.
- See `PROJECT_STRUCTURE.md` for a more detailed description of the CPM module organization.

## Citation

If this code is released together with a paper, add the final bibliographic information here after the paper is accepted/published.
