# Handwritten Digit Recognition with Neural Networks

This project leverages TensorFlow and OpenCV to recognize handwritten digits using a neural network trained on the MNIST dataset. It allows for the training of a new model or the loading of an existing model to predict digits from custom images.

## Features

- **MNIST Dataset:** Train a neural network model on the MNIST dataset of handwritten digits.
- **Custom Image Prediction:** Load custom images of handwritten digits from a folder and predict the digits using the trained model.
- **Interactive Image Viewer:** View images one at a time, along with their predictions, through an interactive interface.

## Prerequisites

Before running this project, ensure you have the following installed:

- Python 3.x
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

You can install the required libraries using the following commands:

```bash
pip install tensorflow opencv-python numpy matplotlib
```

## Usage

### 1. Training a New Model
To train a new model on the MNIST dataset, set the `train_new_model` flag to `True`. The model will be created, trained on the MNIST data, and saved to a file named `handwritten_digits.keras`.

```python
pip install tensorflow opencv-python numpy matplotlib
```

### 2. Predicting Custom Images
To predict custom handwritten digits, place the images in the `digits/` directory. The images should be named `digit1.png`, `digit2.png`, etc. The model will predict the digits and display the predictions interactively.

```python
pip install tensorflow opencv-python numpy matplotlib
```

### 3. Interactive Display
Click to navigate through the images and see predictions in an interactive window. The class `ImageNavigator` provides this functionality by displaying one image at a time and updating the prediction for that image.

## File Structure

- `handwritten_digits.keras`: The trained model file (only if `train_new_model` is `False`).
- `digits/`: A folder containing custom images (`digit1.png`, `digit2.png`, ...) for digit prediction.

## How It Works

### Model Creation:
The model consists of three layers:
- A Flatten layer to reshape the input data.
- Two Dense layers with 128 units and ReLU activation functions.
- An output layer with 10 units and softmax activation for classification into 10 possible digits (0-9).

### Training:
The model is trained using the MNIST dataset for 3 epochs. The optimizer used is Adam, with sparse categorical cross-entropy as the loss function.

### Prediction:
After the model is trained or loaded, custom images are loaded from the `digits/` directory. The model makes predictions and outputs the predicted digit.

### Interactive Viewer:
The `ImageNavigator` class handles displaying images interactively. Clicking on the image will cycle through the predictions of different images in the folder.
