## Protein Sequence Design: Optimizing Stability in Inverse Folding Model
This codebase is developed on top of [MultiFlow (Campbell & Yim et.al, 2024)](https://github.com/jasonkyuyim/multiflow), 
as well as [DRAKES (Wang et.al, 2024)](https://github.com/ChenyuWang-Monica/DRAKES/tree/master). For more detailed 
instructions and explanations about the pre-trained model, datasets, and reward oracles, see the previous two repositories.

### Environment Installation
For the environment installation, please refer to [MultiFlow](https://github.com/jasonkyuyim/multiflow) for details.
```
# Install environment with dependencies.
conda env create -f multiflow.yml

# Activate environment
conda activate multiflow

# Install local package.
# Current directory should have setup.py.
pip install -e .

# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
```

Then install [PyRosetta](https://www.pyrosetta.org/downloads).
```
pip install pyrosetta-installer 
python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'
```

### Fine-Tune to Optimize Protein Stability
```
python finetune_sdpo.py
```

### Evaluate a model
```
python eval.py
```

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
Change the `base_path` in `fmif/eval_finetune.py`, `fmif/train_fmif.py`, `protein_oracle/train_oracle.py` to `BASE_PATH` for saving data and models.

