# Graph Classification
This is a Pytorch implementation of unsupervised graph classification. 

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

### Graph Classification
Just execuate the following command for graph classification task:
```
python main_graph_classification.py
```
Supported datasets:
* `DD`, `PROTEINS`, `NCI1`, `MUTAG`, `COLLAB`, `IMDB-BINARY`, `REDDIT-BINARY`, `REDDIT-MULTI-5K`

### Parameter Settings
| Datasets      | lr        | weight_decay   | layers      | batch_size     | nhid  | temperature |
| :----: | :----: | :----: | :----:	   | :----:       | :----: | :----: |
| MUTAG     | 0.0001     | 0.0001     	 | 5            | 128            | 128     | 0.2		| 
| PROTEINS  | 0.0001     | 0.001          | 5             | 64            | 64      | 0.5		|
| DD	    | 0.0001     | 0.001          | 1             | 128            | 128      | 0.5		|
| NCI1          | 0.0001		| 0.0001          | 6             | 128            | 128     | 1.0			|
| COLLAB            | 0.0001    | 0.0001          | 4              | 128            | 128      | 1.0          |
| IMDB-B      | 0.0001     | 0.1          | 5            | 512            | 256      | 1.0         |
| REDDIT-B      | 0.0001     | 0.0001          | 4             | 128            | 128      | 0.5         |
|REDDIT-M-5K | 0.0001     | 0.0001          | 3             | 128           | 128     | 1.0         |