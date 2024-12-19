import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
from xgboost import XGBClassifier
import shap
import scipy.stats as stats
from collections import defaultdict


class MainClustering:

    def __init__(
        self,
        n_main_clusters,
        min_subcluster_size,
        dataset_success_rate,
        success_column,
        real_world_success_rate,
        n_subclusters=2,
    ):
        self.n_main_clusters = n_main_clusters
        self.min_subcluster_size = min_subcluster_size
        self.dataset_success_rate = dataset_success_rate
        self.success_column = success_column
        self.real_world_success_rate = real_world_success_rate
        self.n_subclusters = n_subclusters
        self.main_clusters = None
        self.significant_features = {}
        self.subcluster_models = None

    def create_main_clusters(self, X):
        print("Creating main clusters...")
        clustering = AgglomerativeClustering(
            n_clusters=self.n_main_clusters, metric="euclidean", linkage="ward"
        )
        self.main_clusters = clustering.fit_predict(X)
        linkage_matrix = linkage(X, method="ward")
        return self.main_clusters, linkage_matrix

    def find_significant_features(self, df, X, feature_names):
        if self.main_clusters is None:
            raise ValueError("Main clusters have not been created yet.")

        # Initialize dictionary to store results
        self.significant_features = {}

        for cluster in range(self.n_main_clusters):
            # Create boolean mask for cluster selection
            cluster_mask = self.main_clusters == cluster
            cluster_size = np.sum(cluster_mask)

            if cluster_size < self.min_subcluster_size:
                continue

            # Use boolean indexing to get cluster data
            cluster_data = df[cluster_mask]
            other_data = df[~cluster_mask]

            # Calculate cluster statistics
            success_rate = cluster_data[self.success_column].mean()
            total_samples = len(cluster_data)
            successful_samples = cluster_data[self.success_column].sum()

            cluster_features = []
            for i, feature in enumerate(feature_names):
                cluster_values = X[cluster_mask, i]
                other_values = X[~cluster_mask, i]

                t_stat, p_value = stats.ttest_ind(
                    cluster_values, other_values, equal_var=True
                )

                if p_value < 0.05:
                    effect_size = (
                        np.mean(cluster_values) - np.mean(other_values)
                    ) / np.std(X[:, i])
                    cluster_features.append(
                        {
                            "feature": feature,
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "effect_size": effect_size,
                            "cluster_success_rate": success_rate,
                            "cluster_total": total_samples,
                            "cluster_success_count": int(successful_samples),
                        }
                    )

            # Sort by effect_size and take the top 4 features
            cluster_features_sorted = sorted(
                cluster_features, key=lambda x: x["effect_size"], reverse=True
            )[:4]

            self.significant_features[cluster] = cluster_features_sorted

        return self.significant_features

    def create_shap_based_subclusters(
        self, processed_df, X, cluster_idx, feature_names
    ):
        """
        Creates subclusters and displays top 4 important features.
        """
        # Get cluster-specific data
        cluster_mask = self.main_clusters == cluster_idx
        cluster_data = X[cluster_mask]
        cluster_success = processed_df[self.success_column].values[cluster_mask]

        if len(cluster_data) < self.min_subcluster_size:
            return None

        # Main cluster statistics
        main_cluster_stats = {
            "total": len(cluster_data),
            "success_count": int(np.sum(cluster_success)),
            "success_rate": cluster_success.mean(),
        }
        print(f"\nMain Cluster {cluster_idx}:")
        print(f"Total: {main_cluster_stats['total']}")
        print(f"Success Count: {main_cluster_stats['success_count']}")
        print(f"Success Rate: {main_cluster_stats['success_rate']:.2%}")

        # Get SHAP values
        model = XGBClassifier(objective="binary:logistic", random_state=42)
        model.fit(cluster_data, cluster_success)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(cluster_data)

        # Get top 4 features by SHAP importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        top_features_idx = np.argsort(feature_importance)[-4:][::-1]

        print("\nTop 4 Important Features:")
        for idx in top_features_idx:
            print(f"- {feature_names[idx]}: {feature_importance[idx]:.4f}")

        # Use top 3 for subclustering (keep this at 3 for manageable subclusters)
        clustering_features_idx = top_features_idx[:3]
        binary_features = np.round(cluster_data[:, clustering_features_idx])

        # Create subclusters using AgglomerativeClustering
        subclustering = AgglomerativeClustering(
            n_clusters=3, metric="euclidean", linkage="ward"  # Force 3 subclusters
        )
        subcluster_labels = subclustering.fit_predict(binary_features)

        # Calculate subcluster statistics
        subcluster_stats = []
        print("\nSubcluster Statistics:")

        for i in range(3):  # Always 3 subclusters
            sub_mask = subcluster_labels == i
            total = np.sum(sub_mask)
            success_count = int(np.sum(cluster_success[sub_mask]))
            success_rate = success_count / total if total > 0 else 0

            # Feature statistics for top 4 features
            feature_stats = {}
            for idx in top_features_idx:  # Use all top 4 for reporting
                feature_name = feature_names[idx]
                feature_values = cluster_data[sub_mask, idx]
                mean_value = np.mean(feature_values)
                mean_shap = np.mean(np.abs(shap_values[sub_mask, idx]))
                feature_stats[feature_name] = {
                    "mean": mean_value,
                    "shap_importance": mean_shap,
                }

            stats = {
                "subcluster": f"{cluster_idx}.{i+1}",
                "total": total,
                "success_count": success_count,
                "success_rate": success_rate,
                "feature_stats": feature_stats,
            }
            subcluster_stats.append(stats)

            print(f"\nSubcluster {cluster_idx}.{i+1}:")
            print(f"Total: {total}")
            print(f"Success Count: {success_count}")
            print(f"Success Rate: {success_rate:.2%}")
            print("Feature Statistics:")
            for feat, info in feature_stats.items():
                print(
                    f"- {feat}: value={info['mean']:.2f}, importance={info['shap_importance']:.4f}"
                )

        return {
            "main_cluster_stats": main_cluster_stats,
            "subcluster_stats": subcluster_stats,
            "subcluster_labels": subcluster_labels,
            "top_features": {
                feature_names[idx]: feature_importance[idx] for idx in top_features_idx
            },
        }
