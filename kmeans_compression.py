import numpy as np
import matplotlib.pyplot as plt
from utils import load_data, plot_progress_kMeans

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

def run_k_means(X, initial_centroids, max_iters=10, plot_progress=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids    
    idx = np.zeros(m)
    
    if plot_progress:
        plt.figure(figsize=(8, 6))

    for i in range(max_iters):
        # Shows 1/10 to 10/10 in terminal
        print(f"K-Means iteration {i+1}/{max_iters}...", end="\r")
        
        idx = find_closest_centroids(X, centroids)
        
        if plot_progress:
            # Shows iteration 1 to 10 in the Plot Title
            plot_progress_kMeans(X, centroids, previous_centroids, idx, K, i+1)
            previous_centroids = centroids
            
        centroids = compute_centroids(X, idx, K)
    
    if plot_progress:
        plt.show()
        
    return centroids, idx

def initialize_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    return X[randidx[:K]]

if __name__ == "__main__":
    print("Step 1: Visualizing K-Means on Sample 2D Dataset")
    try:
        X_sample = load_data() 
        initial_centroids_sample = np.array([[3, 3], [6, 2], [8, 5]])
        K_sample = 3
        max_iters_sample = 10
        run_k_means(X_sample, initial_centroids_sample, max_iters_sample, plot_progress=True)
    except FileNotFoundError:
        print("Error: 'data/ex7_X.npy' not found.")

    print("\nStep 2: Running Image Compression Demo")
    try:
        original_img = plt.imread('bird_small.png')
        if np.max(original_img) > 1.0:
            original_img = original_img / 255.0
            
        X_img = original_img.reshape(-1, 3)
        K_img = 16
        initial_centroids_img = initialize_centroids(X_img, K_img)
        
        centroids_img, idx_img = run_k_means(X_img, initial_centroids_img, max_iters=10, plot_progress=False)
        
        X_compressed = centroids_img[idx_img, :]
        compressed_img = X_compressed.reshape(original_img.shape)
        
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(original_img)
        ax[0].set_title('Original Image')
        ax[1].imshow(compressed_img)
        ax[1].set_title(f'Compressed Image (K={K_img})')
        plt.show()
    except FileNotFoundError:
        print("Error: 'bird_small.png' not found.")
