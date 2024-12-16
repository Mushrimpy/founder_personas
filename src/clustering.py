import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


def create_main_clusters(self, X):
    """Create main cluster groups using hierarchical clustering"""
    print("Creating main clusters...")
    clustering = AgglomerativeClustering(
        n_clusters=self.n_main_clusters, metric="euclidean", linkage="ward"
    )
    main_cluster_labels = clustering.fit_predict(X)
    linkage_matrix = linkage(X, method="ward")

    return main_cluster_labels, linkage_matrix


def analyze_main_clusters(self, df, cluster_labels):
    """Analyze characteristics and success rates of main clusters"""
    print("Analyzing main clusters...")
    results = []

    for cluster in range(self.n_main_clusters):
        mask = cluster_labels == cluster
        cluster_data = df[mask]

        if len(cluster_data) < self.min_subcluster_size:
            continue

        success_rate = cluster_data[self.success_column].mean()
        total_samples = len(cluster_data)
        successful_samples = cluster_data[self.success_column].sum()

        feature_stats = []
        for feature in self.feature_names:
            feature_mean = cluster_data[feature].mean()
            overall_mean = df[feature].mean()
            feature_std = df[feature].std()

            if len(cluster_data) > 1:
                z_score = (feature_mean - overall_mean) / (
                    feature_std / np.sqrt(len(cluster_data))
                )

                if abs(z_score) > 1.96:  # 95% confidence level
                    feature_stats.append(
                        {
                            "feature": feature,
                            "diff": feature_mean - overall_mean,
                            "z_score": z_score,
                        }
                    )

        results.append(
            {
                "cluster_id": cluster + 1,
                "size": total_samples,
                "success_count": successful_samples,
                "success_rate": success_rate,
                "normalized_success_rate": success_rate
                * (self.real_world_success_rate / self.dataset_success_rate),
                "significant_features": sorted(
                    feature_stats, key=lambda x: abs(x["z_score"]), reverse=True
                )[:5],
            }
        )

    return pd.DataFrame(results)


def advanced_clustering(self, X):
    """Combines multiple clustering approaches for robust founder segmentation"""
    results = {}

    # 1. Spectral Clustering for non-linear relationships
    from sklearn.cluster import SpectralClustering

    spectral = SpectralClustering(
        n_clusters=self.n_main_clusters,
        affinity="nearest_neighbors",
        random_state=42,
    )
    results["spectral"] = spectral.fit_predict(X)

    # 2. DBSCAN for density-based clustering
    from sklearn.cluster import DBSCAN

    dbscan = DBSCAN(eps=0.5, min_samples=self.min_subcluster_size)
    results["dbscan"] = dbscan.fit_predict(X)

    # 3. Gaussian Mixture Models for probabilistic clustering
    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=self.n_main_clusters)
    results["gmm"] = gmm.fit_predict(X)

    return results
