## Regulatory DNA Sequence Design
This codebase is developed on top of [MDLM (Sahoo et.al, 2023)](https://github.com/kuleshov-group/mdlm) and 
[DRAKES (Wang et.al, 2024)](https://github.com/ChenyuWang-Monica/DRAKES/tree/master). For more detailed instructions 
and explanations about the pre-trained model, datasets, and reward oracles, see the previous two repositories.

### Environment Installation
```
conda create -n sedd python=3.9.18
conda activate sedd
bash env.sh

# install grelu
git clone https://github.com/Genentech/gReLU.git
cd gReLU
pip install .
```
Note that you need to install `gReLU` package with version 1.0.2.

### Gosai Dataset
The enhancer dataset used for this experiment is provided in `BASE_PATH/mdlm/gosai_data`.

### Fine-tune the pre-trained discrete diffusion model
```
python finetune_sdpo.py
```
Use SDPO to fine-tune a model. 

### Evaluate a model
```
python eval.py
```
Measures the predicted HepG2 activity, ATAC accuracy, JASPAR correlation, Pearson correlation, and approximated log-likelihood,
as discussed in our paper. 

### Generate data from a model
```
python gen_data.py
```

To be used for generating newly labeled samples (additional fine-tuning data).

### Retrain a model on a new dataset
```
python sdpo_retrain.py
```

For fine-tuning a model on additional data.

### Adaptations
Change the `base_path` in `dataloader_gosai.py`, `finetune_reward_bp.py`, `oracle.py`, `train_oracle.py`, `eval.ipynb` to `BASE_PATH` for saving data and models, and change `hydra.run.dir` in `configs_gosai/config_gosai.yaml`.

### Acknowledgement 

This code is based of the [DRAKES repository](https://github.com/ChenyuWang-Monica/DRAKES/tree/master).
* The original dataset is provided by [Gosai et al., 2023](https://www.biorxiv.org/content/10.1101/2023.08.08.552077v1).
* The trained oracle is based on [gReLU](https://genentech.github.io/gReLU/). 
