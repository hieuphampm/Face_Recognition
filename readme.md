# Face Recognition using Metric Learning & Web Deployment

This project demonstrates how to build a **face recognition system** using **Metric Learning** with Triplet Loss. The system learns to map faces into an embedding space where similar faces are close together, and dissimilar faces are far apart.

---

## Overview

### 1. Metric Learning for Face Recognition

We use a **Triplet Network** to train on face datasets:
- Each training step uses a triplet: `Anchor`, `Positive`, and `Negative`.
- The network is trained to:
  - Minimize distance between Anchor and Positive (same identity)
  - Maximize distance between Anchor and Negative (different identities)

> **Model 3 (based on ResNet50)** was selected as the best-performing model based on embedding similarity metrics.

---

### 2. Web Deployment

After training the model, the system can be deployed to the web with the following features:
- Stream a real face from the webcam
- Find and display the most similar face from a reference gallery
- Show prediction results with similarity scores and identity match

---

## Tools for Preparing Dataset

### 1. `rename.py`

This script renames images inside each person's folder to ensure structured and unique filenames.

#### Usage:

```python
rename_images_in_folder("root/childs", prefix="child")
```

- **`root/childs`**: Path to the dataset folder containing subfolders like `A`, `B`, `C`, etc. Each subfolder represents a different identity.
- **`prefix`**: Prefix used when renaming images inside each subfolder.

#### Example:

**Before:**
```
faces/
├── A/
│   ├── image.jpg
├── B/
│   ├── photo.jpg
```

**After running:**
```python
rename_images_in_folder("faces", prefix="A")
```

**After:**
```
faces/
├── A/
│   ├── A_1.jpg
│   ├── A_2.jpg
├── B/
│   ├── B_1.jpg
│   ├── B_2.jpg
```

This step helps to avoid naming conflicts and makes image indexing easier during embedding generation and web retrieval.

---

## Workflow Summary

1. **Prepare dataset** in `root_folder/person_name/image.jpg` format  
2. **Use `rename.py`** to structure image names  
3. **Train the Triplet model (ResNet50)**  
4. **Generate embeddings** for gallery and test images  
5. **Deploy the app** to perform real-time face recognition from webcam  

---

## Note

Because the project is based on personal data, I cannot provide data, please collect it yourself.
