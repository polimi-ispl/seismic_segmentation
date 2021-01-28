# Segmentation of seismic images through CNNs

[Francesco Picetti](mailto:francesco.picetti@polimi.it), Vincenzo Lipari, Paolo Bestagini

### Abstract
Train a CNN architecture to recognize and segment subsurface structures from migrated images.
This repo contains two examples:
1. binary segmentation of salt bodies
2. multiclass segmentation of facies


### Requirements
Conda enviroments can be created via:
```bash
    conda create -f environment.yml
```
Note: this repo relies on pytorch ecosystem.
We adopted [Catalyst](https://github.com/catalyst-team/catalyst) as primary tool for defining the training routines.
It can employs multi-GPU machines, check out their documentation.

