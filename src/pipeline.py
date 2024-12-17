class Pipeline:
    def __init__():
        pass

    def fit_transform(self, df):
        """Run the complete analysis pipeline"""
        print("\nStarting two-stage founder analysis...")

        # Preprocess data
        X, original_df = self.preprocess_data(df)

        # Create main clusters
        main_cluster_labels, linkage_matrix = self.create_main_clusters(X)

        # Analyze clusters
        main_cluster_results = self.analyze_main_clusters(
            original_df, main_cluster_labels
        )
        subcluster_results = self.create_subclusters(
            X, original_df, main_cluster_labels
        )
        print(subcluster_results.shape)
        # Generate summaries
        main_summary, sub_summary = self.generate_summary_tables(
            main_cluster_results, subcluster_results
        )

        # Generate hierarchical view and save all results
        _ = self.generate_hierarchical_table(main_summary, sub_summary)
        self.save_summary_tables(main_summary, sub_summary)

        # Create visualization
        self.visualize_clusters(linkage_matrix)

        print("\nAnalysis complete!")
        return main_cluster_results, subcluster_results, main_cluster_labels
