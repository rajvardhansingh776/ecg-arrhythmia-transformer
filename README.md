# ECG Arrhythmia Detection with Transformers

This repository implements a research pipeline for ECG arrhythmia classification using transformer architectures.

## Features

- Patch-based ECG transformer
- Self-supervised pretraining
- Patient-wise validation
- Cross-dataset benchmarking
- Explainable AI (SHAP + saliency)
- Calibration and reliability analysis
- Ablation studies

## Installation

pip install -r requirements.txt

## Training

python training/train.py

## Cross-dataset experiments

python experiments/cross_dataset.py