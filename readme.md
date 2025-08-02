# Face Recognition using Metric Learning & Web Deployment

This project demonstrates how to build a **face recognition system** using **Metric Learning** with Triplet Loss. The system learns to map faces into an embedding space where similar faces are close together, and dissimilar faces are far apart.

---

## 📌 Overview

### 1. Metric Learning for Face Recognition

We use **Triplet Network** to train on face datasets:
- Each training step uses a triplet: `Anchor`, `Positive`, `Negative`.
- The network is trained to:
  - Minimize distance between Anchor and Positive (same identity)
  - Maximize distance between Anchor and Negative (different identities)

**Model 3 (based on ResNet50)** was selected as the best-performing model based on embedding similarity metrics.

### 2. Web Deployment

After training the model:
- Export the embedding vectors
- Build a web application for:
  - Stream a real face from camera
  - Finding and displaying the most similar face from a gallery
  - Showing prediction results with similarity scores

---

## Tools for Preparing Dataset

### 1. `rename.py`

This script renames images inside each person's folder to ensure unique and structured filenames.

```python
rename_images_in_folder("root/childs", prefix="child")

- root/childs: Path to the dataset folder containing subfolders like A, B, C, etc.

- prefix: Prefix to use when renaming images, e.g., "A" → A_1.jpg, A_2.jpg, etc.