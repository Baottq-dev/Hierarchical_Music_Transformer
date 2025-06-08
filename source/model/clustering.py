import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

def determine_optimal_k(embeddings_array, max_k=10, min_k=2):
    """
    Determines the optimal K for K-means using the Silhouette Score.
    Only considers K if it's less than the number of samples.
    """
    if embeddings_array.shape[0] < min_k:
        print(f"Warning: Number of samples ({embeddings_array.shape[0]}) is less than min_k ({min_k}).")
        return 1

    silhouette_scores = []
    possible_k_values = []

    actual_max_k = min(max_k, embeddings_array.shape[0] -1 if embeddings_array.shape[0] > 1 else 1)

    if actual_max_k < min_k:
        print(f"Warning: Adjusted max_k ({actual_max_k}) is less than min_k ({min_k}).")
        return embeddings_array.shape[0] if embeddings_array.shape[0] >=1 else 1

    for k_val in range(min_k, actual_max_k + 1):
        kmeans = KMeans(n_clusters=k_val, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings_array)
        if len(set(cluster_labels)) > 1 and len(set(cluster_labels)) < embeddings_array.shape[0]:
            score = silhouette_score(embeddings_array, cluster_labels)
            silhouette_scores.append(score)
            possible_k_values.append(k_val)
        else:
            print(f"Could not compute silhouette score for K={k_val}")

    if not silhouette_scores:
        print("Could not determine optimal K using silhouette scores.")
        return min(3, embeddings_array.shape[0]) if embeddings_array.shape[0] > 0 else 1

    optimal_k_index = np.argmax(silhouette_scores)
    optimal_k = possible_k_values[optimal_k_index]
    print(f"Silhouette scores: {silhouette_scores} for K values: {possible_k_values}")
    print(f"Optimal K based on Silhouette Score: {optimal_k}")
    return optimal_k

def cluster_embeddings(embeddings_file, output_file):
    """
    Clusters text embeddings and assigns semantic tokens.
    Args:
        embeddings_file: Path to JSON file containing text embeddings
        output_file: Path to save clustered data
    """
    try:
        with open(embeddings_file, "r") as f:
            data_with_embeddings = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {embeddings_file} not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {embeddings_file}.")
        return

    if not data_with_embeddings:
        print("No data found in the input JSON file.")
        return

    embeddings_list = [item["embedding"] for item in data_with_embeddings]
    if not embeddings_list:
        print("No embeddings found in the data.")
        return

    embeddings_array = np.array(embeddings_list)
    num_samples = embeddings_array.shape[0]

    if num_samples < 2:
        print("Not enough samples to perform clustering.")
        if num_samples == 1:
            cluster_labels = [0]
            final_k = 1
        else:
            return
    elif num_samples <= 5:
        final_k = min(3, num_samples)
        print(f"Small sample size ({num_samples}). Setting K to {final_k}.")
        kmeans = KMeans(n_clusters=final_k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings_array)
    else:
        final_k = determine_optimal_k(embeddings_array, max_k=min(10, num_samples -1), min_k=2)
        if final_k == 1 and num_samples > 1:
            print("Optimal K determination suggested K=1 for multiple samples.")
            final_k = min(3, num_samples // 2 if num_samples // 2 >= 2 else 2) if num_samples >=2 else 1
        
        kmeans = KMeans(n_clusters=final_k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings_array)

    print(f"Clustering into {final_k} clusters.")

    clustered_data = []
    for i, item in enumerate(data_with_embeddings):
        item["semantic_token"] = int(cluster_labels[i])
        clustered_data.append(item)

    try:
        with open(output_file, "w") as f:
            json.dump(clustered_data, f, indent=4)
        print(f"Successfully saved clustered data with semantic tokens to {output_file}")
    except IOError:
        print(f"Error: Could not write to output file {output_file}") 