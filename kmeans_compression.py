import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, plot_progress_kMeans # Import helpers

def find_closest_centroids(X, centroids):
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = np.sum((X[i] - centroids)**2, axis=1)
        idx[i] = np.argmin(distances)
    return idx

def compute_centroids(X, idx, K):
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        points = X[idx == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
    return centroids

# --- MODIFIED RUN_K_MEANS TO SUPPORT VISUALIZATION ---
def run_k_means(X, initial_centroids, max_iters=10, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    
    # Create a figure if we are plotting progress
    if plot_progress:
        plt.figure(figsize=(8, 6))

    for i in range(max_iters):
        # Output progress
        print(f"K-Means iteration {i}/{max_iters-1}...", end="\r")
        
        # Assignment step
        idx = find_closest_centroids(X, centroids)
        
        # Visualization step (only for 2D data)
        if plot_progress:
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids
            
        # Update step
        centroids = compute_centroids(X, idx, K)
    
    if plot_progress:
        plt.show()
        
    return centroids, idx

def initialize_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    return X[randidx[:K]]

if __name__ == "__main__":
    # --- PART 1: 2D SAMPLE DATASET VISUALIZATION ---
    print("Step 1: Visualizing K-Means on Sample 2D Dataset")
    try:
        X_sample = load_data() # Loads from data/ex7_X.npy
        
        # Set initial centroids as per the exercise
        initial_centroids_sample = np.array([[3, 3], [6, 2], [8, 5]])
        K_sample = 3
        max_iters_sample = 10
        
        # Run and Plot
        run_k_means(X_sample, initial_centroids_sample, max_iters_sample, plot_progress=True)
        
    except FileNotFoundError:
        print("Error: 'data/ex7_X.npy' not found. Skip to image compression.")

    # --- PART 2: IMAGE COMPRESSION ---
    print("\nStep 2: Running Image Compression Demo")
    try:
        original_img = plt.imread('bird_small.png')
        # Normalize if necessary (some formats load as 0-255, others 0-1)
        if np.max(original_img) > 1.0:
            original_img = original_img / 255.0
            
        X_img = original_img.reshape(-1, 3)
        K_img = 16
        initial_centroids_img = initialize_centroids(X_img, K_img)
        
        # Run (We don't plot progress for images because thousands of points is too slow)
        centroids_img, idx_img = run_k_means(X_img, initial_centroids_img, max_iters=10, plot_progress=False)
        
        # Reconstruct
        X_compressed = centroids_img[idx_img, :]
        compressed_img = X_compressed.reshape(original_img.shape)
        
        # Display
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(original_img)
        ax[0].set_title('Original Image')
        ax[1].imshow(compressed_img)
        ax[1].set_title(f'Compressed Image (K={K_img})')
        plt.show()
        
    except FileNotFoundError:
        print("Error: 'bird_small.png' not found.")
