import sys
import torch
import numpy as np
import sklearn.datasets
from typing import Any
from dataclasses import dataclass, field


def generate_gaussian_ring(num_clusters=8, n=10000, radius=5, overlapping=0.5, random_state=7):
    """
    Generate a synthetic dataset of 2D Gaussians arranged in a ring.

    Parameters:
    - num_clusters: Number of Gaussian distributions
    - n: Number of total points
    - radius: Radius of the ring where Gaussians are centered
    - overlapping: Controls the overlap (higher value means more overlap)
    - random_state: Random number generator

    Returns:
    - data: Generated 2D points
    - labels: Cluster labels for each point
    """
    np.random.seed(random_state)  # Ensure reproducibility
    points_per_cluster = n // num_clusters

    # Compute standard deviation based on overlap factor
    std_dev = overlapping * radius / num_clusters

    # Generate cluster centers in a circular layout
    angles = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)
    centers = np.array([(radius * np.cos(a), radius * np.sin(a)) for a in angles])

    # Generate data points
    data = []
    labels = []
    for i, center in enumerate(centers):
        cov = np.array([[std_dev, 0], [0, std_dev]])  # Isotropic Gaussian
        points = np.random.multivariate_normal(center, cov, points_per_cluster)
        data.append(points)
        labels.extend([i] * points_per_cluster)

    data = np.vstack(data)
    labels = np.array(labels)

    return data, labels


def generate_truncated_gaussian_ring(num_clusters=8, n=10000, radius=5, cluster_params=None,
                                     overlapping=0.5, variability=0.3, random_state=7):
    """
    Generate a synthetic dataset of 2D Gaussians arranged in a ring with truncated tails.

    Parameters:
    - num_clusters: Number of Gaussian distributions
    - n: Number of total points
    - radius: Mean radius of the ring where Gaussians are centered
    - max_radius: Maximum allowed radius (outer boundary)
    - min_radius: Minimum allowed radius (inner boundary)
    - overlapping: Base standard deviation factor for Gaussian clusters
    - variability: Variation in standard deviation across clusters
    - random_state: Random number generator
    - cluster_params: Precomputed cluster centers & std_devs (for generating a consistent test set)

    Returns:
    - data: Generated 2D points confined within the circular boundary
    - labels: Cluster labels for each point
    """
    np.random.seed(random_state)
    points_per_cluster = n // num_clusters

    if cluster_params is None:
        # Generate cluster centers in a circular layout
        angles = np.linspace(0, 2 * np.pi, num_clusters, endpoint=False)
        centers = np.array([(radius * np.cos(a), radius * np.sin(a)) for a in angles])
        cluster_std_devs = []
    else:
        # Use predefined cluster parameters (for test set)
        centers, cluster_std_devs = cluster_params

    data = []
    labels = []
    for i, center in enumerate(centers):
        # Assign a unique standard deviation to each cluster to vary ring width
        if cluster_params is None:
            cluster_std_dev = overlapping * radius / num_clusters * (1 + variability * (np.random.rand() - 0.5))
            cluster_std_devs.append(cluster_std_dev)
        else:
            cluster_std_dev = cluster_std_devs[i]

        accepted_points = []
        while len(accepted_points) < points_per_cluster:
            # Generate a batch of points
            batch_size = points_per_cluster - len(accepted_points)
            points = np.random.multivariate_normal(center, np.diag([cluster_std_dev] * 2), batch_size)

            # Compute Mahalanobis distance for truncation
            normalized_distances = np.linalg.norm((points - center) / cluster_std_dev, axis=1)
            # Keep only points within [-1.96σ, 1.96σ] (95% confidence interval)
            valid_points = points[normalized_distances <= 3]
            # Add valid points, ensuring exactly `points_per_cluster` per cluster
            accepted_points.extend(valid_points[:points_per_cluster - len(accepted_points)])

        data.append(np.array(accepted_points))
        labels.extend([i] * points_per_cluster)

    data = np.vstack(data)
    labels = np.array(labels)

    return data, labels, (centers, cluster_std_devs)


@dataclass
class InputData:
    """Collects data in the format required by the MSSP solver"""
    X: np.array
    Xval: np.array
    name: str
    limits: Any = field(init=False)
    n_features: int = field(init=False)

    def __post_init__(self):
        # Shuffle dataset
        indexes = np.arange(len(self.X))
        np.random.seed(7)
        np.random.shuffle(indexes)
        self.X = self.X[indexes]
        self.Xval = self.Xval[indexes]
        self.n_features = self.X.shape[1]


class DataLoader:
    """Class used to load or generate datasets"""

    def __init__(self, name=None):
        """
        Initialize DataLoader class
        :param name: Dataset name (If known, otherwise create a new temporal dataset)
        """
        self.X, self.Xval, self.Y, self.names = np.zeros(0), np.zeros(0), np.zeros(0), None
        self.name = name

        self.modelType = "NN"
        if hasattr(self, f'{name}'):
            method = getattr(self, f'{name}')
            method()
        else:
            sys.exit('The provided dataset name does not exist')

        self.dataset = InputData(X=self.X, Xval=self.Xval, name=self.name)

    def sr(self, n=10000):  # Swiss roll
        np.random.seed(7)
        X, _ = sklearn.datasets.make_swiss_roll(n_samples=n, noise=0.3, random_state=7)
        srx = torch.tensor(X, dtype=torch.float)
        self.X = torch.cat((srx[:, 0].reshape(-1, 1), srx[:, 2].reshape(-1, 1)), 1) / 10
        X, _ = sklearn.datasets.make_swiss_roll(n_samples=n, noise=0.3, random_state=3)
        srx = torch.tensor(X, dtype=torch.float)
        self.Xval = torch.cat((srx[:, 0].reshape(-1, 1), srx[:, 2].reshape(-1, 1)), 1) / 10

    def gr(self, n=10000):  # Swiss roll
        np.random.seed(7)
        X, _ = generate_gaussian_ring(num_clusters=8, n=n, radius=5, overlapping=0.5, random_state=7)
        self.X = torch.tensor(X, dtype=torch.float)
        X, _ = generate_gaussian_ring(num_clusters=8, n=n, radius=5, overlapping=0.5, random_state=3)
        self.Xval = torch.tensor(X, dtype=torch.float)

    def grT(self, n=10000):  # Swiss roll
        np.random.seed(7)
        X, _, train_params = generate_truncated_gaussian_ring(num_clusters=8, n=n, radius=5, overlapping=0.5,
                                                              variability=0.5, random_state=7)
        self.X = torch.tensor(X, dtype=torch.float)
        X, _, _ = generate_truncated_gaussian_ring(cluster_params=train_params, random_state=3)
        self.Xval = torch.tensor(X, dtype=torch.float)


# import matplotlib.pyplot as plt
# data0, labels0, tr_params = generate_truncated_gaussian_ring(num_clusters=8, n=10000, radius=5,
#                                                 overlapping=0.5, variability=0.5, random_state=7)
#
# plt.figure(figsize=(6, 6))
# plt.scatter(data0[:, 0], data0[:, 1], edgecolors='k', s=10)
# plt.xlim(-6, 6)
# plt.ylim(-6, 6)
# plt.gca().set_aspect('equal', adjustable='box')
# # plt.title("Truncated Gaussian Ring Dataset")
# plt.show()
