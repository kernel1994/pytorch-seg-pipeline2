# Image semantic segmentation pipeline with PyTorch

## Abstract
This code is a pipeline for semantic segmentation task.
KFold is adopted as default cross validation strategy.

## Pipeline
The pipeline consists of five parts:
- data preparation
- model creation
- model training & val
- data prediction
- dice evaluation

Configure: `cfg.py`

Entry: `main.py`

Evaluated by [EvaluateSegmentation](https://github.com/Visceral-Project/EvaluateSegmentation)

## TODO
- [ ] tensorboard
