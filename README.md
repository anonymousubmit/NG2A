# NG2A
This is a PyTorch implementation of our proposed NG2A algorithm.

## Requirements
* python3.8
* pytorch==1.11.0
* torch-scatter==2.0.9
* torch-sparse==0.6.13
* torch-cluster==1.6.0
* torch-geometric==2.0.4

## Datasets
Datasets used in graph classification and node classification will be downloaded automatically via Pytorch Geometric when running the codes. You can refer [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) for more details. For graph transfer learning, the datasets can be found [here](https://github.com/snap-stanford/pretrain-gnns#dataset-download), which is about 2.5GB. After having downloaded it, you need to unzip it, and put it under `transfer-learning/dataset/`.

## Quick Start:

### Graph Classification
Just execuate the following command for graph classification task:
```
python main_graph_classification.py
```
Supported datasets:
* `DD`, `PROTEINS`, `NCI1`, `MUTAG`, `COLLAB`, `IMDB-BINARY`, `REDDIT-BINARY`, `REDDIT-MULTI-5K`

For detailed parameter settings, please refer to `README.md` in `./unsupervised-graph`.

### Node Classification
Just execuate the following command for node classification task:
```
python main_node_classification.py
```
Supported datasets:
* `WikiCS`, `Computers`, `Photo`, `CS`, `Physics`

For detailed parameter settings, please refer to `README.md` in `./unsupervised-node`.

### Transfer Learning
Just execuate the following command for pre-training:
```
python pretraining.py
```
Then, execuate the following command for finetune:
```
python finetune.py
```
Supported datasets:
* Pretrain datasets: `Zinc`
* Finetune datasets: `BBBP`, `Tox21`, `ToxCast`, `SIDER`, `ClinTox`, `MUV`, `HIV`, `BACE`

For detailed parameter settings, please refer to `README.md` in `./transfer-learning`.
