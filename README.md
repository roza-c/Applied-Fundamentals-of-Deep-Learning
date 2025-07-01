# Applied Fundamentals of Deep Learning

This repository contains a series of labs I completed as part of a course on the applied fundamentals of deep learning. Each lab is a self-contained Jupyter Notebook that explores different concepts and architectures in deep learning using PyTorch.

## Labs Completed

Below is a summary of the labs included in this repository.

---

### [Lab 1: PyTorch and ANNs](Lab1_PyTorch_and_ANNs.ipynb)

This introductory lab served as a warm-up to get me used to the PyTorch environment. It covered the foundational concepts required for the course, including a review of Python and its relevant libraries for deep learning.

**Key skills I developed:**
* Performing basic tensor operations in PyTorch.
* Loading and handling data within the PyTorch framework.
* Configuring and building Artificial Neural Networks (ANNs).
* Training ANNs using PyTorch's training loop.
* Evaluating and comparing the performance of different ANN configurations.

---

### [Lab 2: Cats vs Dogs](Lab2_Cats_vs_Dogs.ipynb)

In this lab, I trained a Convolutional Neural Network (CNN) to classify images as either "cat" or "dog". While the neural network code was provided, the focus was on understanding the training process and key machine learning concepts.

**Key concepts I learned:**
* The high-level structure of a machine learning training loop.
* The distinction between training, validation, and test datasets.
* Recognizing the concepts of overfitting and underfitting.
* Investigating the impact of hyperparameters like learning rate and batch size on training success.
* Comparing the performance of an ANN (Multi-Layer Perceptron) with a CNN.

---

### [Lab 3: Gesture Recognition using Convolutional Neural Networks](Lab3_Gesture_Recognition.ipynb)

This lab challenged me to build and train a CNN for hand gesture classification. For this lab, no starter code was provided, and I was expected to adapt code from earlier labs and lectures to complete the tasks.

**Key skills I developed:**
* Loading and splitting data effectively for training, validation, and testing.
* Training a Convolutional Neural Network from scratch.
* Applying the technique of transfer learning to improve model performance.

---

### [Lab 4: Data Imputation using an Autoencoder](Lab4_Data_Imputation.ipynb)

In this lab, I built and trained an autoencoder to fill in missing values in the UCI Adult Data Set. The model learned to predict missing features based on the available information for each data entry.

**In the process, I learned to:**
* Clean and preprocess a mix of continuous and categorical data for machine learning.
* Implement an autoencoder that can handle both continuous and one-hot encoded categorical inputs.
* Tune the hyperparameters of an autoencoder for optimal performance.
* Use baseline models as a benchmark to interpret the performance of my autoencoder.

---

### [Lab 5: Spam Detection](Lab5_Spam_Detection.ipynb)

In this assignment, I built a Recurrent Neural Network (RNN) to classify SMS text messages as "spam" or "not spam". I worked with text data and learned the fundamentals of sequence modeling.

**In this assignment, I learned to:**
* Clean and process text data for machine learning applications.
* Understand and implement a character-level Recurrent Neural Network.
* Utilize `torchtext` to build and batch data for RNN models.
* Comprehend the specifics of batching for recurrent networks and implement it using `torchtext`.

---

## How to Use

Each lab is contained within its own Jupyter Notebook (`.ipynb`) file. To run the labs, a Python environment with PyTorch and other relevant data science libraries (such as NumPy, Pandas, and Matplotlib) is required.