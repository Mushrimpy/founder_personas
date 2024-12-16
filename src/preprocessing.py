import pandas as pd


def prepare_dataset(self, df):
    print("Preprocessing data...")

    # Verify success column exists
    if self.success_column not in df.columns:
        raise ValueError(f"Success column '{self.success_column}' not found in dataset")

    # Calculate dataset success rate
    original_success_rate = df[self.success_column].mean()
    print(f"Original success rate: {original_success_rate:.1%}")

    # Split data into success and failure samples
    success_samples = df[df[self.success_column] == 1]
    failure_samples = df[df[self.success_column] == 0]

    return success_samples, failure_samples


def balance_classes(self, df, success_samples, failure_samples):

    # Calculate the required number of failure samples to achieve target success rate
    n_success = len(success_samples)
    target_n_failure = int(
        n_success * (1 - self.real_world_success_rate) / self.real_world_success_rate
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
    self.dataset_success_rate = df[self.success_column].mean()
    print(f"Dataset success rate: {self.dataset_success_rate:.1%}")


def process_features(self, df):
    feature_map = {
        "FALSE": 0,
        "TRUE": 1,
        "nope": 0,
        "unit": 1,
        "multi": 2,
    }

    df = pd.DataFrame({col: df[col].map(feature_map) for col in df.columns})

    # Select numerical features
    numerical_features = df.select_dtypes(include=["float64", "int64"]).columns
    numerical_features = numerical_features[numerical_features != self.success_column]

    self.feature_names = numerical_features.tolist()
    print(f"Selected {len(self.feature_names)} features")

    # Handle missing values and scale
    X = self.imputer.fit_transform(df[numerical_features])
    X = self.scaler.fit_transform(X)

    return X, df


def preprocess_data(self, df):
    """Preprocess the data and calculate dataset statistics"""
    success_samples, failure_samples = self.prepare_dataset(df)

    balanced_df = self.balance_classes(df, success_samples, failure_samples)

    X, processed_df = self.process_features(balanced_df)

    return X, processed_df