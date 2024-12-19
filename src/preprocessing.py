import pandas as pd
import numpy as np


class DataPreprocessor:
    def __init__(self, real_world_success_rate, success_column, imputer, scaler):
        self.real_world_success_rate = real_world_success_rate
        self.success_column = success_column
        self.imputer = imputer
        self.scaler = scaler

    def _prepare_dataset(self, df):
        print("Preprocessing data...")

        # Verify success column exists
        if self.success_column not in df.columns:
            raise ValueError(
                f"Success column '{self.success_column}' not found in dataset"
            )

        # Calculate dataset success rate
        original_success_rate = df[self.success_column].mean()
        print(f"Original success rate: {original_success_rate:.1%}")

        # Split data into success and failure samples
        success_samples = df[df[self.success_column] == 1]
        failure_samples = df[df[self.success_column] == 0]

        return success_samples, failure_samples

    def _balance_classes(self, df, success_samples, failure_samples):

        # Calculate the required number of failure samples to achieve target success rate
        n_success = len(success_samples)
        target_n_failure = int(
            n_success
            * (1 - self.real_world_success_rate)
            / self.real_world_success_rate
        )

        # If more failure samples are needed, perform random resampling with replacement
        if target_n_failure > len(failure_samples):
            resampled_failures = failure_samples.sample(
                n=target_n_failure,
                replace=True,
                random_state=42,
            )
            df = pd.concat([success_samples, resampled_failures])
        else:
            # If enough failure samples exist, randomly select the required amount
            sampled_failures = failure_samples.sample(
                n=target_n_failure,
                replace=False,
                random_state=42,
            )
            df = pd.concat([success_samples, sampled_failures])

        # Calculate dataset success rate
        df_success_rate = df[self.success_column].mean()
        print(f"Dataset success rate: {df_success_rate:.1%}")

        return df, df_success_rate

    def _process_features(self, df):
        num_map = {
            "FALSE": 0,
            "TRUE": 1,
            "nope": 0,
            "unit": 1,
            "multi": 2,
            "undisclosed": np.nan,
            "nope": 0,
            "l20": 0.4,
            "50_150": 0.6,
            "150_500": 0.8,
            "g500": 1,
        }

        for column in df.columns:
            df[column] = df[column].apply(lambda x: num_map[x] if x in num_map else x)
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Select numerical features
        numerical_features = df.select_dtypes(include=["float64", "int64"]).columns
        numerical_features = numerical_features[
            numerical_features != self.success_column
        ]

        self.feature_names = [
            feature
            for feature in numerical_features.tolist()
            if feature not in ["founder_uuid", "name", "org_name", "persona"]
        ]

        print(f"Selected {len(self.feature_names)} features")

        # Let the imputer handle all missing values
        X = self.imputer.fit_transform(df[numerical_features])
        X = self.scaler.fit_transform(X)

        return (
            X,
            df.drop(columns=["founder_uuid", "name", "org_name", "persona"]),
            self.feature_names,
        )

    def preprocess_data(self, df):
        success_samples, failure_samples = self._prepare_dataset(df)

        balanced_df, df_success_rate = self._balance_classes(
            df, success_samples, failure_samples
        )

        X, processed_df, feature_names = self._process_features(balanced_df)

        return X, processed_df, feature_names, df_success_rate
