# Project 2: Handwritten Digit Recognition (MNIST)

## Overview

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) using TensorFlow and Keras to recognize handwritten digits from the MNIST dataset.

- **Dataset**: MNIST — 70,000 grayscale images of handwritten digits (0 to 9)
- **Model**: Sequential CNN with two convolutional layers and two fully connected layers
- **Objective**: Accurately classify digits from 0 to 9 using deep learning

## Features

- Image preprocessing and normalization
- One-hot encoding for output labels
- CNN architecture using `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense` layers
- Uses ReLU and Softmax activations
- Model training with Adam optimizer
- Evaluation on unseen test data
- Visualization of sample predictions

## Requirements

- **Python 3.x**
- **TensorFlow**
- **Matplotlib**

Install dependencies using:

```bash
pip install tensorflow matplotlib
```

## How to Run

1. Save the Python script (e.g., `digit_recognition.py`).
2. Run the script in terminal or any Python IDE:

```bash
python digit_recognition.py
```

### The script will:

- Train the model for 5 epochs
- Evaluate it on the test set
- Display predictions for sample test images

## Output

- Displays test accuracy (typically over 98%)
- Shows 10 test images with their predicted digits using Matplotlib
