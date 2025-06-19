# Discrete Diffusion Trajectory Alignment via Stepwise Decomposition

The repository contains the code for the `SDPO` method presented in the paper: Discrete Diffusion Trajectory Alignment
via Stepwise Decomposition.
`SDPO` is a fine-tuning method for the reward optimization or alignment of discrete diffusion models.

## Data and Model Weights

We use the data and checkpoints from the [DRAKES repository](https://github.com/ChenyuWang-Monica/DRAKES/tree/master).

## Regulatory DNA Sequence Design

We fine-tune a pre-trained discrete diffusion model to generate DNA sequences that exhibit high enhancer activity, while still remaining natural-like. The detailed code and instructions are in `SDPO_dna/`. 

## Protein Sequence Design: Optimizing Stability in Inverse Folding Model

We fine-tune a pre-trained inverse protein folding model to generate protein sequences that demonstrate high stability, while still remaining natural and of the correct structure. The code and instructions are in `SDPO_protein/`.
