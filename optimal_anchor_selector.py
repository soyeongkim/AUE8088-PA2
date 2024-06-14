import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pathlib import Path

# Load your dataset
def load_labels(path):
    labels = []
    for label_file in path.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 8:  # Ensure there are enough parts
                    x_min = float(parts[4])
                    y_min = float(parts[5])
                    x_max = float(parts[6])
                    y_max = float(parts[7])
                    width = x_max - x_min
                    height = y_max - y_min
                    labels.append([width, height])
    return np.array(labels)

# Normalize labels to feature map sizes
# def normalize_labels(labels, img_size, feature_size):
#     return labels * (feature_size / img_size)

def normalize_labels(labels, ratio):
    return labels * ratio

# Run K-means to get anchors
def get_anchors(labels, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(labels)
    return kmeans.cluster_centers_, kmeans.labels_

# Visualize anchors
def plot_anchors(anchors, labels, cluster_labels):
    plt.figure(figsize=(10, 10))
    unique_labels = np.unique(cluster_labels)
    for i in unique_labels:
        cluster_points = labels[cluster_labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
    plt.scatter(anchors[:, 0], anchors[:, 1], c='red', label='Anchors', marker='X', s=200)
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('K-means clustering')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main code
label_path = Path('/home/ailab/AUE8088-PA2/datasets/nuscenes_det2d/train/label_2')
labels = load_labels(label_path)
if labels.size == 0:
    raise ValueError("No labels found. Please check the label path and files.")

# Define feature map sizes and image size
# img_size = 416
# feature_sizes = [32, 16, 8]  # Example feature sizes for P3, P4, P5

# all_anchors = []
# for ratio in ratios:
#     norm_labels = normalize_labels(labels, ratio)
#     anchors = get_anchors(labels, num_clusters=3)
#     all_anchors.append(anchors)
#     plot_anchors(anchors, norm_labels)

all_anchors = []
anchors, cluster_labels = get_anchors(labels, num_clusters=3)
all_anchors.append(anchors)
plot_anchors(anchors, labels, cluster_labels)

# Reshape anchors to fit YOLOv5 format
all_anchors = np.array(all_anchors).reshape(3, -1, 2)

print("Anchors:", all_anchors)
