# Image Classification on MNIST Dataset

## Objective
In this assignment, you will build an image classification model using the MNIST dataset. The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). You will perform the following tasks:
1. Load and preprocess the dataset.
2. Build and train a neural network for digit classification.
3. Analyze if overfitting is occurring and apply techniques to minimize it.


```python
import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Explore the dataset
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Testing images shape: {test_images.shape}")
print(f"Testing labels shape: {test_labels.shape}")
```

