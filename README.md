# AI-Programming-with-Python-Nanodegree
# Pre-trained Image Classifier to Identify Dog Breeds

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
- Pre-trained models (VGG)
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
