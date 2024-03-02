# Convergent Learning for Class Imbalance: A Unified Approach to Long-Tail Recognition in Image Classification

This project explores the challenges of class imbalance in image classification and introduces an integrated approach that combines metric learning, cross-entropy, and contrastive learning techniques to enhance feature separability and ensure equitable class representation.

## Abstract
Class imbalances within datasets can skew the performance of machine learning models toward majority classes and neglect minority ones. This study introduces methods to harmonize the learning process across all classes, thereby improving the model's efficacy.

## Installation

Before running the project, install the required packages:

```
torch
torchvision
matplotlib
seaborn
sklearn
numpy
```

## Dataset
The dataset.py script is used to create imbalanced versions of CIFAR10, CIFAR100, and custom datasets. It supports various transformations and allows for a flexible approach to dataset preparation.

### Usage
```
from dataset import ImbalanceCIFAR10, ImbalanceCIFAR100

# Example for CIFAR10
dataset = ImbalanceCIFAR10(root='data', imb_ratio=0.1, transform=transforms)
```
## Models
The project includes implementations of ResNet architectures. The models.py script provides classes such as SupConResNet and SupCEResNet with support for contrastive learning and cross-entropy respectively.

### Usage
```from models import SupConResNet, SupCEResNet

# Example for SupConResNet
model = SupConResNet(name='resnet50', head='mlp', feat_dim=128, num_classes=10)```

## How to Cite Please cite this work as:
Bishoy Soliman Hanna, "Convergent Learning for Class Imbalance: A Unified Approach to Long-Tail Recognition in Image Classification," Worcester Polytechnic Institute, Feb 2024.


