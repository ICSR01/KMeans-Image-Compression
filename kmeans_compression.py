import numpy as np
import matplotlib.pyplot as plt

def find_closest_centroids(X, centroids):
    """
    Assigns every data point to the nearest centroid.
    """
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        # Calculate squared Euclidean distance to each centroid
        distances = np.sum((X[i] - centroids)**2, axis=1)
        # Assign to the index of the minimum distance
        idx[i] = np.argmin(distances)
        
    return idx

def compute_centroids(X, idx, K):
    """
    Recalculates centroids by taking the mean of all points assigned to each cluster.
    """
    m, n = X.shape
    centroids = np.zeros((K, n))
    
    for k in range(K):
        points = X[idx == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
            
    return centroids

def run_k_means(X, initial_centroids, max_iters=10):
    """
    The main loop that runs the K-Means algorithm.
    """
    centroids = initial_centroids
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, centroids.shape[0])
    return centroids, idx

def initialize_centroids(X, K):
    """
    Randomly selects K data points as initial centroids.
    """
    randidx = np.random.permutation(X.shape[0])
    return X[randidx[:K]]

# --- Example Usage for Image Compression ---
if __name__ == "__main__":
    # 1. Load an image (assuming a local file named 'bird.png')
    # Use plt.imread or any sample image array
    try:
        original_img = plt.imread('bird_small.png')
        
        # 2. Reshape image to (N_pixels, 3) where N = width * height
        X_img = original_img.reshape(-1, 3)
        
        # 3. Run K-Means to find 16 colors (K=16)
        K = 16
        initial_centroids = initialize_centroids(X_img, K)
        centroids, idx = run_k_means(X_img, initial_centroids, max_iters=10)
        
        # 4. Map each pixel to its centroid color
        X_compressed = centroids[idx, :]
        compressed_img = X_compressed.reshape(original_img.shape)
        
        # 5. Display results
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(original_img)
        ax[0].set_title('Original Image')
        ax[1].imshow(compressed_img)
        ax[1].set_title(f'Compressed Image (K={K})')
        plt.show()
        
    except FileNotFoundError:
        print("Image file not found. Place 'bird_small.png' in the directory to run the demo.")
