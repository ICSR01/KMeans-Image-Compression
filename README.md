# KMeans-Image-Compression

Implementation of the K-Means clustering algorithm from scratch in Python for lossy image compression and color-space optimization.

---

## 🚀 Quick Links
* **Core Implementation**: [kmeans_compression.py](./kmeans_compression.py) (Manual implementation of the K-Means logic)
* **Demonstration**: [C3_W1_KMeans_Assignment.ipynb](./C3_W1_KMeans_Assignment.ipynb) (Notebook with visualizations)

---

## Table of Contents
1. [Description of Technology](#description-of-technology)
2. [Description of the Process](#description-of-the-process)
3. [Visual Results](#visual-results)
4. [Project Background & Motivation](#project-background--motivation)
5. [Problem Statement](#problem-statement)
6. [Intended Use](#intended-use)
7. [Challenges & Limitations](#challenges--limitations)
8. [Credits](#credits)

---

## Description of Technology
* **Python**: The primary language for algorithm development and data handling.
* **NumPy**: Utilized for high-performance vectorized linear algebra.
* **Matplotlib**: Employed for 2D/3D data visualization and image rendering.

### Why?
The choice of **NumPy** was a mechanical necessity for this project. The K-Means algorithm requires calculating Euclidean distances across thousands of pixels ($128 \times 128$) and multiple centroids. Standard Python loops would lead to significant latency; NumPy’s vectorization allows for simultaneous broadcasting, ensuring the model converges in real-time.

---

## Description of the Process
The project followed a modular pipeline:
1.  **Centroid Initialization**: Selecting initial points to serve as the "center" of clusters.
2.  **Assignment (Expectation)**: Calculating distances and assigning each pixel to the nearest centroid.
3.  **Update (Maximization)**: Re-computing centroids based on the mean of all assigned points.
4.  **Convergence**: Iterating the process until the centroids stabilize.

**Core Code Reference:** The manual implementation of Exercise 1 (Centroid Assignment) and Exercise 2 (Centroid Update) can be found in [kmeans_compression.py](./kmeans_compression.py).

### Why?
I utilized an **iterative optimization approach** to minimize the "distortion" (cost function). By isolating the "Assignment" and "Update" logic into separate functions, the code remains readable and easily testable against synthetic datasets before being applied to high-dimensional image data.

---

## Visual Results
The algorithm was tested by compressing a 24-bit color image into a 4-bit representation (16 colors).

| Original Image (thousands of colors) | Compressed Image (K=16) |
| :---: | :---: |
| ![Original Image Placeholder](./bird_small.png) | ![Compressed Image Placeholder](./bird_compressed.png) |

> **Note:** The compressed image retains the essential visual structure and color depth of the original while significantly reducing the data required to represent each pixel.

---

## Project Background & Motivation
### How the project came about
This project was developed as part of the **Machine Learning Specialization** by DeepLearning.AI on Coursera. It served as a practical exploration of unsupervised learning architectures.

### The Motivation
The motivation was to move beyond simple classification and understand how machine learning can discover latent structures within unlabelled data. Mastering clustering is a foundational step toward more complex tasks like anomaly detection and data preprocessing.

---

## Problem Statement
**What problem it hopes to solve:** High-resolution images consume significant storage and bandwidth. This project addresses **data footprint reduction**. By clustering similar colors and representing them with a single centroid, we can reduce an image's color palette (e.g., from thousands of colors to just 16), achieving substantial compression with minimal loss in visual quality.

---

## Intended Use
* **Technical Portfolio**: To demonstrate proficiency in building machine learning algorithms from the ground up.
* **Data Preprocessing Reference**: The clustering logic implemented here can be adapted for feature engineering in complex datasets where grouping similar observations is required.
* **Academic Demonstration**: A clear example of how mathematical optimization (minimizing Euclidean distance) translates into a practical software solution.

---

## Challenges & Limitations
* **Challenges**: A significant hurdle was the **Local Optima** problem. Since K-Means is sensitive to initial centroid positions, poor initialization can lead to suboptimal clustering. I addressed this by implementing a random initialization strategy to increase the likelihood of finding a global optimum.
* **Limitations**: The algorithm requires the user to pre-define the number of clusters ($K$). Furthermore, K-Means assumes that clusters are spherical and of similar size, which may not accurately reflect every data distribution.

---

## Credits
* **DeepLearning.AI**: For providing the instructional framework and dataset.
* **Andrew Ng**: For the theoretical guidance on optimization and vectorization.
* **Coursera**: The platform providing the development environment.
