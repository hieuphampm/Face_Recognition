# Triplet Network Model Comparison for Face Recognition

This project involves building and comparing three deep learning models for learning image embeddings using **Triplet Loss**. The aim is to maximize the similarity between Anchor/Positive (A/P) pairs while minimizing similarity between Anchor/Negative (A/N) pairs.

## Files

- `train_model_1.ipynb`: First model – basic CNN for embedding extraction.
- `train_model_2.ipynb`: Second model – deeper CNN with additional regularization.
- `train_model_3.ipynb`: Third model – based on a pretrained ResNet50 backbone.

---

## Model Architectures

### Model 1: Simple CNN
A basic convolutional neural network that includes:
- 2 convolutional layers
- Max pooling
- Fully connected projection layer

**Performance:**
- A/P similarity: **0.9057**
- A/N similarity: **0.2550**

---

### Model 2: Deep CNN with Dropout
An improved CNN that introduces:
- 3 convolutional layers
- Dropout regularization
- Slightly deeper network

**Performance:**
- A/P similarity: **0.6626**
- A/N similarity: **-0.0033**

> This model suffers from poor separation between positive and negative samples.

---

### Model 3: ResNet50 Backbone
A powerful model that leverages:
- Pretrained **ResNet50**
- Custom embedding head
- Batch normalization and improved projection

**Performance:**
- A/P similarity: **0.9732**
- A/N similarity: **0.2578**

> **Best-performing model** in terms of learning effective embeddings.

---

## Final Model Selection

After evaluation, **Model 3 (ResNet50)** was chosen as the best model due to its:
- Highest A/P similarity (strong positive clustering)
- Highest A/N separation
- Transfer learning advantages

---

## How to Run
Create an root folder, in the root folder will have child folder (with each folder is a label of person)

Open the corresponding Jupyter Notebook (`train_model_X.ipynb`) in Jupyter Lab or Google Colab and run the cells sequentially.
