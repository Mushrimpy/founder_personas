import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
from copy import deepcopy

from .preprocessing import DataPreprocessor
from .clustering import MainClustering


class FounderAnalyzer:
    def __init__(
        self,
        n_main_clusters=8,
        min_subcluster_size=20,
        real_world_success_rate=0.019,
        success_column="success",
    ):
        self.n_main_clusters = n_main_clusters
        self.min_subcluster_size = min_subcluster_size
        self.real_world_success_rate = real_world_success_rate
        self.success_column = success_column
        self.dataset_success_rate = None
        self.imputer = SimpleImputer(strategy="mean")
        self.scaler = StandardScaler()

        self.cluster_trees = {}
        self.subcluster_paths = {}
        self.main_cluster_labels = None
        self.X = None

        self.preprocessor = DataPreprocessor(
            real_world_success_rate=self.real_world_success_rate,
            success_column=self.success_column,
            imputer=self.imputer,
            scaler=self.scaler,
        )

        self.main_clustering = MainClustering(
            n_main_clusters=self.n_main_clusters,
            min_subcluster_size=self.min_subcluster_size,
            dataset_success_rate=self.dataset_success_rate,
            success_column=self.success_column,
            real_world_success_rate=self.real_world_success_rate,
        )
