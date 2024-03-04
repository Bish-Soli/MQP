# Convergent Learning for Class Imbalance: A Unified Approach to Long-Tail Recognition in Image Classification

This project explores the challenges of class imbalance in image classification and introduces an integrated approach that combines metric learning, cross-entropy, and contrastive learning techniques to enhance feature separability and ensure equitable class representation.

## Abstract
Class imbalances within datasets can skew the performance of machine learning models toward majority classes and neglect minority ones. This study introduces methods to harmonize the learning process across all classes, thereby improving the model's efficacy.

## Installation

Before running the project, install the required packages:

```
pip install -r requirements.txt
```

## Dataset
The dataset.py script is used to create imbalanced versions of CIFAR10, CIFAR100, and custom datasets. It supports various transformations and allows for a flexible approach to dataset preparation.


## Custom Dataset

To run this code on a Custom Dataset, the directory structure needs to follow that of Pytorch ImageFolder.

## Models
The project includes implementations of ResNet architectures. The models.py script provides classes such as SupConResNet and SupCEResNet with support for contrastive learning and cross-entropy respectively.

## Hardware

We used Google Colab A100 GPU.

## Code Execution

In order to run the code on CIFAR10/100, please folow the below format:

```
!python main.py --model resnet18 --dataset cifar10  --method CE --epoch 100 --learning_rate 0.1 --batch_size 32 --debug 0

```
For the Supervised Contrastive + Cross Entropy, you can use the below format:

```
!python main.py --model resnet18 --dataset cifar10  --method SupCon --epoch 100 --learning_rate 0.1 --batch_size 32 --debug 0 --alpha 0.5 --beta 0.5
```

For the Custom Dataset, please follow the below format:

```
!python main.py --model resnet34 --data_folder /content/dataset --dataset path  --method CE --epoch 100  --learning_rate 0.1 --batch_size 32  --image_size 64 
```
For the Supervised Contrastive + Cross Entropy on the customer dataset, you can use the below format:

```
!python main.py --model resnet18 --data_folder /content/dataset --dataset path  --method SupCon --epoch 100  --learning_rate 0.1 --batch_size 32  --image_size 64 --alpha 0.5 --beta 0.5
```
Our experiment file on Google Colab for CIFAR10/100 is throw Colab_CIFAR10_&_CIFAR100.ipynb, also for the Custom Dataset is throw Colab_Custom_Dataset.ipynb 


## How to Cite Please cite this work as:

Bishoy Soliman Hanna, "Convergent Learning for Class Imbalance: A Unified Approach to Long-Tail Recognition in Image Classification," Worcester Polytechnic Institute, Feb 2024.


