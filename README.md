# Sonnet: A self-guided ordinal regression neural network for segmentation and classification of nuclei in large-scale multi-tissue histology images

## Overview

This repository contains a tensorflow implementation of SONNET: a self-guided ordinal regression neural network that performs simultaneously nuclei segmentation and classification. By introducing a distance decreasing discretization strategy, the network can detect nuclear pixels (inner pixels) and high uncertainty regions (outer pixels). The self-guided training strategy is applied to high uncertainty regions to improve the final outcome. 

The repository includes:
- Source code of SONNET.
- Training code for SONNET.
- Datasets employed in the SONNET paper.
- Pretrained weights for the SONNET encoder.
- Evaluation on nuclei segmentation metrics (DICE, AJI, DQ, SQ, PQ) and nuclei classification metrics (Fd and F1 scores).

## Installation
```
conda create --name sonnet python=3.6
conda activate sonnet
pip install -r requirements.txt
```
