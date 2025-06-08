"""
Clustering module for AMT model.
"""

import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import os
from ..data_processing.midi_processor import analyze_midi_file

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

class MIDIClusterer:
    """
    Clustering for MIDI files.
    """
    def __init__(self, n_clusters: int = 10):
        """
        Initialize clusterer.
        Args:
            n_clusters: Number of clusters
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_centers = None
        self.cluster_labels = None
    
    def extract_features(self, midi_file: str) -> np.ndarray:
        """
        Extract features from MIDI file.
        Args:
            midi_file: Path to MIDI file
        Returns:
            Feature vector
        """
        # Analyze MIDI file
        analysis = analyze_midi_file(midi_file)
        
        # Extract features
        features = [
            analysis["note_density"],
            analysis["velocity_mean"],
            analysis["velocity_std"],
            analysis["note_range"],
            analysis["tempo_mean"],
            analysis["tempo_std"]
        ]
        
        return np.array(features)
    
    def fit(self, midi_files: List[str]):
        """
        Fit clustering model.
        Args:
            midi_files: List of MIDI file paths
        """
        # Extract features
        features = []
        for midi_file in midi_files:
            try:
                feature_vector = self.extract_features(midi_file)
                features.append(feature_vector)
            except Exception as e:
                print(f"Error processing {midi_file}: {e}")
        
        # Convert to numpy array
        X = np.array(features)
        
        # Fit KMeans
        self.kmeans.fit(X)
        self.cluster_centers = self.kmeans.cluster_centers_
        self.cluster_labels = self.kmeans.labels_
    
    def predict(self, midi_file: str) -> int:
        """
        Predict cluster for MIDI file.
        Args:
            midi_file: Path to MIDI file
        Returns:
            Cluster label
        """
        # Extract features
        features = self.extract_features(midi_file)
        
        # Predict cluster
        return self.kmeans.predict(features.reshape(1, -1))[0]
    
    def get_cluster_center(self, cluster_id: int) -> np.ndarray:
        """
        Get cluster center.
        Args:
            cluster_id: Cluster ID
        Returns:
            Cluster center vector
        """
        return self.cluster_centers[cluster_id]
    
    def get_cluster_files(self, midi_files: List[str], cluster_id: int) -> List[str]:
        """
        Get files in cluster.
        Args:
            midi_files: List of MIDI file paths
            cluster_id: Cluster ID
        Returns:
            List of MIDI files in cluster
        """
        cluster_files = []
        for i, midi_file in enumerate(midi_files):
            if self.cluster_labels[i] == cluster_id:
                cluster_files.append(midi_file)
        return cluster_files

def cluster_midi_files(
    midi_files: List[str],
    n_clusters: int = 10,
    output_file: str = None
) -> Dict[str, Any]:
    """
    Cluster MIDI files.
    Args:
        midi_files: List of MIDI file paths
        n_clusters: Number of clusters
        output_file: Path to output JSON file
    Returns:
        Dictionary containing clustering results
    """
    # Initialize clusterer
    clusterer = MIDIClusterer(n_clusters=n_clusters)
    
    # Fit clustering model
    clusterer.fit(midi_files)
    
    # Get cluster assignments
    cluster_assignments = {}
    for i, midi_file in enumerate(midi_files):
        cluster_id = clusterer.cluster_labels[i]
        if cluster_id not in cluster_assignments:
            cluster_assignments[cluster_id] = []
        cluster_assignments[cluster_id].append(midi_file)
    
    # Create results dictionary
    results = {
        "n_clusters": n_clusters,
        "cluster_centers": clusterer.cluster_centers.tolist(),
        "cluster_assignments": cluster_assignments
    }
    
    # Save to file if specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Set paths
    midi_dir = "data/midi"
    output_file = "data/processed/clusters.json"
    
    # Get MIDI files
    midi_files = []
    for root, _, files in os.walk(midi_dir):
        for file in files:
            if file.endswith('.mid'):
                midi_files.append(os.path.join(root, file))
    
    # Cluster MIDI files
    results = cluster_midi_files(midi_files, n_clusters=10, output_file=output_file)
    
    # Print results
    print(f"Number of clusters: {results['n_clusters']}")
    for cluster_id, files in results['cluster_assignments'].items():
        print(f"\nCluster {cluster_id}:")
        print(f"Number of files: {len(files)}")
        print("Sample files:")
        for file in files[:5]:
            print(f"- {os.path.basename(file)}") 