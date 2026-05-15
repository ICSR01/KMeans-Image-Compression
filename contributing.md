# Contributing to the K-Means Neuro-Research Pipeline

Thank you for your interest in contributing! This project started as a tool for **K-Means Image Compression**, but the mathematical core we've built is designed to be the foundation for **Neuroimaging** and **Computational Neuroscience** workflows.

We are looking for contributors to help bridge the gap between "Pixels" and "Brain Data."

---

## 🔬 From Pixels to Brain Data: How You Can Help
The logic you see in `kmeans_compression.py` is highly versatile. We want to adapt this project to handle specific neuroscientific tasks:

### 1. Neuroimaging (MRI) Segmentation
Instead of clustering RGB values (3 dimensions), we want to cluster voxel intensities from MRI scans. 
* **Goal**: Modify the pipeline to segment Grey Matter, White Matter, and CSF.
* **Relevant Code**: Adapting `find_closest_centroids` to handle 3D volume data.

### 2. Electrophysiological Spike Sorting
The same Euclidean distance math used for compression can be used to "sort" neural spikes.
* **Goal**: Cluster neural waveforms based on their principal components (PCA).
* **Relevant Code**: Using the initialization logic in `initialize_centroids` for high-dimensional waveform features.

### 3. Optimization for Large Neuro-datasets
Neuroscience datasets are often massive. We are looking for:
* **Vectorization**: Improvements to the NumPy broadcasting in our current loops.
* **Memory Management**: Logic to handle large `.npy` or `.nii` (NIfTI) files without crashing the environment.

---

## How to Contribute

### Reporting Issues
If you encounter a bug or if the algorithm fails on a specific type of neural dataset:
1. Open a **GitHub Issue**.
2. State the dimensions of your data (e.g., $128 \times 128 \times 128$ voxels).
3. Provide the error log from the terminal.

### Suggesting Enhancements
We welcome "Proposals" for scientific rigor. If you have experience with **K-Means++** or **Silhouette Analysis** (to find the best $K$ for brain clusters), please open an issue to discuss the implementation.

---

## Technical Guidelines
To keep the project research-grade and readable for both engineers and clinicians:

* **Interpretability**: Use descriptive variable names. Use `voxel_intensity` if working with MRI, or `spike_waveform` for electrophysiology.
* **Code Structure**: Keep the modular split between the math (`kmeans_compression.py`) and the visualization (`utils.py`).
* **Reproducibility**: Ensure all contributions can be run with the dependencies listed in `requirements.txt`.

---

## Ethics & Open Science
* **No Patient Data**: Strictly avoid uploading raw or identifiable patient data. Use only de-identified or synthetic neural datasets for testing.
* **Transparency**: Document any data normalization (like Z-scoring) performed before the clustering step, as this impacts biological interpretation.

---
*By contributing, you agree that your work will be licensed under the project's Open Source license to support the advancement of neuro-informatics.*
