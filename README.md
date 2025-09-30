# Be Reliable: An Interpretable Attribute-Oriented Representation Learning Framework

## Overview

This repository contains the implementation of four variants: DRLNet, GSpNet, Hyp-DRLNet, Hyp-GSpNet.
The framework is designed for robust and generalizable node clustering across various graph datasets.

## Configuration

All hyperparameters, including the number of layers and other model-specific settings, are defined in the `config` directory:

- `layer.yaml`: Specifies the number of layers.

## Datasets

The datasets used in our experiments include **Chameleon**, **Squirrel**, **Film**, **Wiki**, **Texas**, and **Wisconsin**. Please place all dataset files in the `data` directory.

You can download the datasets from the following [Google Drive link](https://drive.google.com/drive/folders/1DQnq850E5xl_PDpV7XR_H904GxRFEW1K).

## Running the Code

To run the four model variants, use the following commands:

```bash
python cluster_DRLNet.py         # DRLNet: Denoising Representation Learning Network
python cluster_GSpNet.py         # GSpNet: Group Sparsity-enhanced Denoising Network
python cluster_Hyp-DRLNet.py     # Hyp-DRLNet: Hypergraph-guided Denoising Network
python cluster_Hyp-GSpNet.py     # Hyp-GSpNet: Group Sparsity-enhanced Hypergraph-guided Denoising Network
