# image-colorization-autoencoder
A deep learning project based on a convolutional autoencoder that colorizes grayscale images using the CIFAR-10 dataset. The model is built with TensorFlow/Keras and demonstrates image reconstruction from low-dimensional latent representations.

# Image Colorization using Convolutional Autoencoder

This project implements a **Convolutional Autoencoder** using TensorFlow/Keras to colorize grayscale images from the CIFAR-10 dataset.

---

##  Project Overview

The model learns to map grayscale images back to their original RGB format using deep learning.

- Input: Grayscale images (32x32x1)
- Output: Color images (32x32x3)
- Model: Convolutional Autoencoder

---

##  Model Architecture

- Encoder: 3 Conv2D layers with downsampling
- Latent space: Dense vector (256 dimensions)
- Decoder: Conv2DTranspose layers for upsampling

---

##  Dataset

- CIFAR-10 dataset (built-in in Keras)
- 60,000 images (32x32 RGB)

---

##  Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

##  Training

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Epochs: 30
- Batch size: 32

---

## 🖼 Results

The model successfully reconstructs colorized images from grayscale inputs.

Example output is saved in:
