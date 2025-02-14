# AI-Programming-with-Python-Nanodegree
This repository contains my submissions for the [nanodegree program AI Programming with Python](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089) offered by [Udacity](https://www.udacity.com/).

## Pre-trained Image Classifier to Identify Dog Breeds

This project focuses on improving Python programming skills by utilizing a pre-trained image classifier to identify dog breeds. The project involves evaluating different convolutional neural network (CNN) architectures, including AlexNet, VGG, and ResNet, to determine their accuracy and efficiency. By comparing classification results and runtime, the goal is to gain insights into the trade-offs between computational cost and model performance. The main task is to apply Python skills to work with the classifier, not to develop a new model.

## Developing an Image Classifier

### Project Overview
This project involves implementing an image classifier using PyTorch to identify dog breeds. The focus is on developing deep learning skills while working with pre-trained models. The project is divided into two parts:
1. Building and training the classifier in a Jupyter notebook.
2. Converting the classifier into a command-line application.

The dataset for this project can be downloaded from [here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html).

### Technologies Used
- Python
- PyTorch
- Convolutional Neural Networks (CNNs)
- Pre-trained models (VGG, AlexNet, ResNet)
- Jupyter Notebook

## How to Use

### Part 1: Model Training
1. Work through the Jupyter notebook to implement the classifier.
2. Train the network using the dataset and save the model checkpoint.
3. Monitor training loss, validation loss, and accuracy.

### Part 2: Command-line Application
The project includes three Python scripts:
- **train.py**: Trains a new model and saves it as a checkpoint.
- **predict.py**: Uses the trained model to classify images.
- **Model_utils.py**: Includes functions used in `train.py` and `predict.py` (e.g., load data, create model, train model, load checkpoint, save checkpoint, process image, predict).

#### Training a Model
**Basic usage:**
```bash
python train.py data_directory
```
#### Making Predictions
**Basic usage:**
```bash
python predict.py /path/to/image checkpoint
```

## Badges
[Pre-trained Image Classifier to Identify Dog Breeds](https://cdn.getblueshift.com/pictures/194761/content/p1-completed-aws-winter.jpg?bsft_aaid=8d7e276e-4a10-41b2-8868-423fe96dd6b2&bsft_eid=83ae9158-3512-5bdb-e66d-8ca11524a58c&utm_campaign=sch_600_ndxxx_aws-ai-ml-summer-project-completed&utm_source=blueshift&utm_medium=email&utm_content=sch_600_ndxxx_aws-ai-ml-summer-project-1-completed&bsft_clkid=62fc5f88-6ec5-4dfe-b48b-559b387068cd&bsft_uid=46fd4b94-3f21-480b-b92f-c33234f80ab4&bsft_mid=0f3f5105-3532-434c-acf8-fac53105686b&bsft_txnid=5e2a7e4a-06c8-4057-91f0-3429de5349d2&bsft_mime_type=html&bsft_ek=2024-11-13T14%3A12%3A15Z&bsft_lx=3&bsft_tv=10)

[Developing an Image Classifier](https://cdn.getblueshift.com/pictures/196996/content/p2-aws-winter.jpg?bsft_eid=b844b7e7-a9a6-d4b5-92e7-9106d74e43d8&utm_campaign=sch_600_ndxxx_aws-ai-ml-summer-project-completed&utm_source=blueshift&utm_medium=email&utm_content=sch_600_ndxxx_aws-ai-ml-summer-project-2-completed&bsft_clkid=e9acd82d-07c8-4306-8f16-fc91d23e7fc7&bsft_uid=46fd4b94-3f21-480b-b92f-c33234f80ab4&bsft_mid=71ccf9cb-fb19-43a4-ac95-b59d48fcc6a9&bsft_txnid=debc3964-4b14-4114-a0ff-7c4657171368&bsft_mime_type=html&bsft_ek=2025-02-05T09%3A40%3A34Z&bsft_lx=3&bsft_tv=7&bsft_aaid=8d7e276e-4a10-41b2-8868-423fe96dd6b2)

## Certificate
(www.udacity.com/certificate/e/25ea19ba-8b40-11ef-8fbd-07035cf0b613)

