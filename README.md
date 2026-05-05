# CMD-PLA

Official project page and code release for the CMD-PLA paper.

> Paper title, author list, venue, and DOI/arXiv information are left as editable fields because they should match the final manuscript metadata exactly.

## Paper

- **Paper:** [CMD_PLA.pdf](docs/CMD_PLA.pdf)
- **Title:** TODO: replace with the full paper title
- **Authors:** TODO: replace with author names
- **Venue:** TODO: replace with conference/journal/preprint information
- **Code:** [lihaoaaa882-oss/CMD-PLA](https://github.com/lihaoaaa882-oss/CMD-PLA)

## Overview

CMD-PLA is a deep learning framework for protein-ligand affinity prediction. The repository contains the training, preprocessing, prediction, graph-construction, and CMD molecular geometry components used by the paper.

The implementation includes:

- a protein-ligand graph dataset pipeline in `CMD_PLA_main/dataset.py`;
- the main hybrid graph model in `CMD_PLA_main/HG.py`;
- training and evaluation entry points in `CMD_PLA_main/train.py` and `CMD_PLA_main/predict.py`;
- CMD molecular geometry utilities under `CMD_PLA_main/CMD_core/`;
- configuration files under `CMD_PLA_main/config_hg/`.

## Repository Structure

```text
.
+-- CMD_PLA_main/
|   +-- CMD_core/              # CMD molecular geometry module
|   +-- config_hg/             # training configuration
|   +-- log/                   # logging helpers
|   +-- dataset.py             # graph dataset and data loader
|   +-- HG.py                  # main hybrid graph model
|   +-- preprocessing.py       # data preprocessing utilities
|   +-- train.py               # training script
|   +-- predict.py             # evaluation / prediction script
+-- docs/
|   +-- CMD_PLA.pdf            # paper PDF
+-- CITATION.cff
+-- requirements.txt
+-- README.md
```

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n cmd-pla python=3.10
conda activate cmd-pla
pip install -r requirements.txt
```

Install PyTorch and PyTorch Geometric versions that match your CUDA version. See the official PyTorch and PyG installation instructions if your CUDA toolkit differs from the default environment used in the paper.

## Data

Place the processed datasets under `CMD_PLA_main/data/` or update `data_root` in `CMD_PLA_main/config_hg/TrainConfig.json`.

The current scripts expect dataset splits such as:

```text
data/
+-- PDBbind/
+-- CASF_2013/
+-- CASF_2016/
+-- train.csv
+-- valid.csv
+-- test.csv
+-- test_2013.csv
+-- test_2016.csv
```

Large datasets and trained checkpoints should usually be released through GitHub Releases, Zenodo, Google Drive, or another external storage service, then linked here.

## Training

Edit `CMD_PLA_main/config_hg/TrainConfig.json` to select the dataset path, GPU id, batch size, number of epochs, and model-saving directory.

```bash
cd CMD_PLA_main
python train.py
```

## Prediction

After preparing the data and checkpoints, run:

```bash
cd CMD_PLA_main
python predict.py
```

Before publishing the repository, replace any local absolute checkpoint paths in `predict.py` with paths relative to the repository or document where users can download the trained models.

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{cmdpla2026,
  title   = {TODO: Full paper title},
  author  = {TODO: Author list},
  journal = {TODO: Venue or preprint server},
  year    = {2026},
  url     = {TODO: DOI, arXiv, or repository URL}
}
```

The same placeholder metadata is also available in `CITATION.cff` for GitHub's citation panel.

## License

TODO: choose and add a license file before making the repository public. Common choices for academic code releases include MIT, Apache-2.0, and BSD-3-Clause.

## Contact

TODO: add the corresponding author's email address or GitHub profile.
