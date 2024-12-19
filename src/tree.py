import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy import stats
import xgboost as xgb
import shap
from collections import defaultdict
from openai import OpenAI


class TreeParser:
    def extract_tree_paths(self, xgb_model, feature_names):
        """Extract decision paths from XGBoost trees"""
        tree_paths = []
        booster = xgb_model.get_booster()

        # Convert tree to text representation and parse paths
        for tree_idx in range(booster.num_boosted_rounds()):
            tree_dump = booster.get_dump(dump_format="text")[tree_idx]
            paths = self._parse_tree_dump(tree_dump, feature_names)
            tree_paths.extend(paths)

        return tree_paths

    def _parse_tree_dump(self, tree_dump, feature_names):
        """Parse XGBoost tree dump to extract decision rules"""
        paths = []
        lines = tree_dump.split("\n")
        current_path = []

        def process_node(line):
            if "[f" in line:  # Internal node
                feature_idx = int(line[line.find("f") + 1 : line.find("<")])
                threshold = float(line[line.find("<") + 1 : line.find("]")])
                return f"{feature_names[feature_idx]} < {threshold:.2f}"
            return None

        # Simplified path extraction - you might want to make this more robust
        for line in lines:
            if line.strip():
                node_rule = process_node(line)
                if node_rule:
                    current_path.append(node_rule)
                elif "leaf" in line:
                    paths.append(current_path.copy())
                    current_path.pop() if current_path else None

        return paths
