import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from xgboost import XGBClassifier
import shap


class MainClustering:

    def __init__(
        self,
        n_main_clusters,
        min_subcluster_size,
        dataset_success_rate,
        success_column,
        real_world_success_rate,
    ):
        self.n_main_clusters = n_main_clusters
        self.min_subcluster_size = min_subcluster_size
        self.dataset_success_rate = dataset_success_rate
        self.success_column = success_column
        self.real_world_success_rate = real_world_success_rate

    def create_main_clusters(self, X):
        """Create main cluster groups using hierarchical clustering"""
        print("Creating main clusters...")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_main_clusters, metric="euclidean", linkage="ward"
        )
        main_cluster_labels = clustering.fit_predict(X)
        linkage_matrix = linkage(X, method="ward")

        return main_cluster_labels, linkage_matrix

    def analyze_main_clusters(self, df, cluster_labels, feature_names, df_success_rate):
        """Analyze characteristics and success rates of main clusters using SHAP"""

        print("Analyzing main clusters...")
        results = []

        # Iterate through each cluster
        for cluster in range(self.n_main_clusters):
            mask = cluster_labels == cluster
            cluster_data = df[mask]

            if len(cluster_data) < self.min_subcluster_size:
                continue

            # Cluster statistics
            success_rate = cluster_data[self.success_column].mean()
            total_samples = len(cluster_data)
            successful_samples = cluster_data[self.success_column].sum()

            # Train a model on the cluster-specific data
            model = XGBClassifier()  # Use a simple classifier
            model.fit(cluster_data[feature_names], cluster_data[self.success_column])

            # Calculate SHAP values for this cluster
            explainer = shap.TreeExplainer(model)
            cluster_shap_values = explainer.shap_values(cluster_data[feature_names])

            # Aggregate SHAP values to find the most important features
            feature_importance = []
            for idx, feature in enumerate(feature_names):
                mean_shap = np.abs(cluster_shap_values[:, idx]).mean()
                cluster_mean = cluster_data[feature].mean()
                global_mean = df[feature].mean()

                feature_importance.append(
                    {
                        "feature": feature,
                        "shap_value": mean_shap,
                        "cluster_mean": cluster_mean,
                        "global_mean": global_mean,
                        "diff": cluster_mean - global_mean,
                    }
                )

            # Sort features by SHAP values
            significant_features = sorted(
                feature_importance, key=lambda x: x["shap_value"], reverse=True
            )[:5]

            # Store cluster analysis results
            results.append(
                {
                    "cluster_id": cluster + 1,
                    "size": total_samples,
                    "success_count": successful_samples,
                    "success_rate": success_rate,
                    "normalized_success_rate": success_rate
                    * (self.real_world_success_rate / df_success_rate),
                    "important_features": significant_features,
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
