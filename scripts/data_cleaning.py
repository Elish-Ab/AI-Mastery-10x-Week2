import pandas as pd
import numpy as np
from scipy import stats

# loads data in csv form from notebooks folder
def load_dataset(data_path):

    return pd.read_csv(data_path)

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    # Replace missing values with the mean or mode
    df['TCP'].fillna(df['TCP'].mean(), inplace=True)
    df['RTT'].fillna(df['RTT'].mean(), inplace=True)
    df['Throughput'].fillna(df['Throughput'].mean(), inplace=True)

    return df

def handle_outliers(df):
    """Handle outliers in the dataset using Z-score method."""
    z_scores = stats.zscore(df[['TCP', 'RTT', 'Throughput']])
    abs_z_scores = np.abs(z_scores)

    # Define a threshold to identify outliers
    threshold = 3
    outlier_rows = (abs_z_scores > threshold).any(axis=1)

    # Replace outliers with the mean value
    df.loc[outlier_rows, ['TCP', 'RTT', 'Throughput']] = df[['TCP', 'RTT', 'Throughput']].mean()

    return df