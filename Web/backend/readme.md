
Each subfolder under `root_folder` should be named after a unique label (e.g., `person_1`, `person_2`, etc.), and must contain at least two images to generate triplets (anchor, positive, negative).

---

## ✅ Usage

Use this folder structure when preparing data for training any of the models (Model 1, 2, or 3). The model will:
- Iterate through each label folder
- Sample image triplets (anchor, positive, negative) during training
- Learn embeddings using Triplet Loss

---

## Same Structure for Inference

When evaluating or retrieving similar images:
- Keep the gallery (reference images) organized in the same folder structure.
- Test images can be provided separately and will be compared against the gallery embeddings.

---

## Tip

Make sure image filenames do not repeat across different folders, and that image extensions are consistent (`.jpg`, `.png`, etc.).
