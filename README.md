# Convolutional Neural Network (CNN) for Handwritten Digit Classification

This project implements a deep Convolutional Neural Network (CNN) to classify images from the MNIST dataset, a standard benchmark for image recognition. The model achieved high accuracy, demonstrating effective feature extraction and generalization capabilities for grayscale image data.

## Key Features

* **Deep Learning Model:** Implementation of a custom, multi-layer CNN architecture.
* **High Performance:** Achieved a **99%+ accuracy** and F1-score on the independent test dataset.
* **Learning Analysis:** Analysis of learning curves to confirm efficient training and validate against overfitting.

## Technology Stack

* **Framework:** **TensorFlow** (as the backend engine).
* **API:** **Keras** (for high-level model construction and training).
* **Data Handling:** **NumPy** for numerical operations and preprocessing.
* **Visualization:** **Matplotlib** for plotting learning curves.
* **Evaluation:** **Scikit-learn** for final performance metrics (Accuracy, F1-score).

## Model Architecture

The CNN was designed with a progression of layers optimized for image features:

1.  **Input Layer:** $28 \times 28$ pixel grayscale images.
2.  **Convolutional Blocks (3 Layers):** Sequential `Conv2D` layers with ReLU activation, progressively increasing the feature map count (32 $\rightarrow$ 64 filters).
3.  **Pooling:** Two `MaxPooling2D` layers ($2 \times 2$) to reduce dimensionality and increase feature robustness.
4.  **Dense Layers:** A `Flatten` layer transitions to a 1D vector, feeding into a 64-unit ReLU dense layer, and culminating in a 10-unit **Softmax** output layer for classification.

## Results

| Metric | Test Set Result | 
| :--- | :--- | 
| **Accuracy** | 0.9923 ($99.23\%$) | 
| **F1 Score (Weighted)** | 0.9923 | 

The learning curves (Epoch vs. Loss) showed consistent convergence, with the validation loss tracking closely to the training loss, indicating strong generalization and **no significant overfitting** after 10 epochs.
