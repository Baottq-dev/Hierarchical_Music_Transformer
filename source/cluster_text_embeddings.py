import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt # For elbow method visualization if needed

def determine_optimal_k(embeddings_array, max_k=10, min_k=2):
    """
    Determines the optimal K for K-means using the Silhouette Score.
    Only considers K if it's less than the number of samples.
    """
    if embeddings_array.shape[0] < min_k:
        print(f"Warning: Number of samples ({embeddings_array.shape[0]}) is less than min_k ({min_k}). Cannot determine optimal K effectively. Returning 1.")
        return 1 # or embeddings_array.shape[0] if each sample is a cluster

    silhouette_scores = []
    possible_k_values = []

    # Ensure K is not greater than n_samples - 1 for silhouette score
    # and not greater than n_samples for KMeans itself.
    # K must be >= 2 for silhouette score.
    actual_max_k = min(max_k, embeddings_array.shape[0] -1 if embeddings_array.shape[0] > 1 else 1)

    if actual_max_k < min_k:
        print(f"Warning: Adjusted max_k ({actual_max_k}) is less than min_k ({min_k}) after considering sample size. Using K={embeddings_array.shape[0]} or 1.")
        return embeddings_array.shape[0] if embeddings_array.shape[0] >=1 else 1

    for k_val in range(min_k, actual_max_k + 1):
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings_array)
        # Silhouette score requires at least 2 labels and at most n_samples - 1 labels.
        if len(set(cluster_labels)) > 1 and len(set(cluster_labels)) < embeddings_array.shape[0]:
            score = silhouette_score(embeddings_array, cluster_labels)
            silhouette_scores.append(score)
            possible_k_values.append(k_val)
        else:
            # Cannot compute silhouette score if only one cluster is formed or all points are in their own cluster
            # For very small datasets, this might happen.
            print(f"Could not compute silhouette score for K={k_val} (num_labels={len(set(cluster_labels))})")
            # silhouette_scores.append(-1) # Assign a low score

    if not silhouette_scores: # If no valid K found for silhouette score
        print("Could not determine optimal K using silhouette scores. Defaulting to a small K or number of samples.")
        # Fallback: if K=2 wasn't possible, and we have few samples, maybe each is its own cluster or one cluster
        return min(3, embeddings_array.shape[0]) if embeddings_array.shape[0] > 0 else 1

    optimal_k_index = np.argmax(silhouette_scores)
    optimal_k = possible_k_values[optimal_k_index]
    print(f"Silhouette scores: {silhouette_scores} for K values: {possible_k_values}")
    print(f"Optimal K based on Silhouette Score: {optimal_k}")
    return optimal_k

if __name__ == "__main__":
    input_embeddings_path = "./data/output/text_embeddings.json"
    output_clustered_data_path = "./data/output/clustered_text_data.json"

    try:
        with open(input_embeddings_path, "r") as f:
            data_with_embeddings = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {input_embeddings_path} not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_embeddings_path}.")
        exit()

    if not data_with_embeddings:
        print("No data found in the input JSON file.")
        exit()

    embeddings_list = [item["embedding"] for item in data_with_embeddings]
    if not embeddings_list:
        print("No embeddings found in the data.")
        exit()

    embeddings_array = np.array(embeddings_list)

    # Determine the number of clusters (K)
    # For a small dataset like the sample (5 items), K will be small.
    # The paper mentions 128 semantic tokens for music, but not explicitly for text-derived ones.
    # Let's try to determine K, but cap it for small sample sizes.
    num_samples = embeddings_array.shape[0]
    # Optimal K determination might not be very meaningful for 5 samples.
    # We will set K to a small number, e.g., 3, if num_samples is small, or try to determine it.
    if num_samples < 2:
        print("Not enough samples to perform clustering.")
        # Assign a default cluster label if only one sample
        if num_samples == 1:
            cluster_labels = [0]
            final_k = 1
        else:
            exit()
    elif num_samples <= 5: # For very small samples, set K directly or use a small number
        final_k = min(3, num_samples) # e.g., K=3 for 5 samples, K=2 for 2 samples
        print(f"Small sample size ({num_samples}). Setting K to {final_k}.")
        kmeans = KMeans(n_clusters=final_k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings_array)
    else:
        # For larger datasets, we could use the elbow method or silhouette analysis
        # For now, let's use the determine_optimal_k function (which uses silhouette)
        final_k = determine_optimal_k(embeddings_array, max_k=min(10, num_samples -1), min_k=2)
        if final_k == 1 and num_samples > 1: # if optimal_k returned 1 for multiple samples, it means it failed. Use a small k.
            print("Optimal K determination suggested K=1 for multiple samples, which is not ideal. Setting K to a small default (e.g., 3 or num_samples/2).")
            final_k = min(3, num_samples // 2 if num_samples // 2 >= 2 else 2) if num_samples >=2 else 1
        
        kmeans = KMeans(n_clusters=final_k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings_array)

    print(f"Clustering into {final_k} clusters.")

    # Add cluster labels (semantic tokens) to the data
    clustered_data = []
    for i, item in enumerate(data_with_embeddings):
        item["semantic_token"] = int(cluster_labels[i]) # K-means labels are 0-indexed
        clustered_data.append(item)

    try:
        with open(output_clustered_data_path, "w") as f:
            json.dump(clustered_data, f, indent=4)
        print(f"Successfully saved clustered data with semantic tokens to {output_clustered_data_path}")
    except IOError:
        print(f"Error: Could not write to output file {output_clustered_data_path}.")


