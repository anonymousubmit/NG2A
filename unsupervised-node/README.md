# Node Classification
This is a Pytorch implementation of unsupervised node classification. 

## Requirements
* python3.8
* pytorch==1.11.0
* torch-scatter==2.0.9
* torch-sparse==0.6.13
* torch-cluster==1.6.0
* torch-geometric==2.0.4

## Datasets
Datasets used in graph classification and node classification will be downloaded automatically via Pytorch Geometric when running the codes. You can refer [here](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) for more details.

## Quick Start:

### Node Classification
Just execuate the following command for node classification task:
```
python main_node_classification.py
```
Supported datasets:
* `WikiCS`, `Computers`, `Photo`, `CS`, `Physics`

### Parameter Settings
| Datasets      | lr        | weight_decay   | layers      | batch_size     | nhid  | temperature |
| :----: | :----: | :----: | :----:	   | :----:       | :----: | :----: |
| WikiCS     | 0.0001     | 0.001     	 | 1           | full           | 1024     | 0.5	| 
| Computers  | 0.0001     | 0.001          | 2             | full            | 1024      | 0.5		|
| Photo 	 | 0.0001     | 0.01          | 2             | full            | 1024      | 0.5		|
| Co.CS      | 0.0001		| 0.001          | 1             | full            | 1024     | 0.5			|
| Co.Physics | 0.0001    | 0.001          | 1           | full           | 1024      | 0.5          |
