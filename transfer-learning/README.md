# Transfer Learning
This is a Pytorch implementation of graph transfer learning on moleculer property prediction.

## Requirements
* python3.8
* pytorch==1.11.0
* torch-scatter==2.0.9
* torch-sparse==0.6.13
* torch-cluster==1.6.0
* torch-geometric==2.0.4
* tqdm==4.64.0
* rdkit==2022.03.3

## Datasets
The datasets can be found [here](https://github.com/snap-stanford/pretrain-gnns#dataset-download), which is about 2.5GB. After having downloaded it, you need to unzip it, and put it under `dataset/`.

## Quick Start:

### Transfer Learning
Just execuate the following command for pre-training:
```
python pretraining.py
```

This will pre-train model using dataset specified in `--dataset`, and the pre-trained model will be saved to the folder specified in `--output_model_file`.

Then, execuate the following command for finetune:
```
python finetune.py
```

This will finetune pre-trained model specified in `--input_model_file` using dataset specified in `--dataset`. Our pre-trained model is avaliable in `./init_weights/model_100.pth`.

Supported datasets:
* Pretrain datasets: `Zinc`
* Finetune datasets: `BBBP`, `Tox21`, `ToxCast`, `SIDER`, `ClinTox`, `MUV`, `HIV`, `BACE`

## Parameter Settings
We use the default parameters specified in the code.

## Acknowledgements
The implementation is based on the codes in Hu et al. [Strategies for Pre-training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns).