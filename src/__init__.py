import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
import matplotlib.pyplot as plt
from copy import deepcopy


class FounderAnalyzer:
    def __init__(
        self,
        n_main_clusters=5,
        min_subcluster_size=15,
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

    from preprocessing import preprocess_data
