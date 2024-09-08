import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st
from data_cleaning import load_dataset, handle_missing_values, handle_outliers

# Load the dataset
file_path = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\loaded_data.csv"
data = load_dataset(file_path)

# Data Cleaning
data = handle_missing_values(data)
data = handle_outliers(data)

print(data.columns)
# Task 3.1: Aggregate Customer Information
agg_data = data.groupby('Customer_ID').agg({
    'TCP': 'mean',
    'RTT': 'mean',
    'Throughput': 'mean',
    'Handset_Type': 'first'
}).reset_index()

# Task 3.2: List Top and Bottom Values
top_values = data.nlargest(5, ['TCP', 'RTT', 'Throughput'])
bottom_values = data.nsmallest(5, ['TCP', 'RTT', 'Throughput'])

# Task 3.3: Analyze Data Distribution
throughput_distribution = data.groupby('Handset_Type')['Throughput'].mean()
print("Mean Throughput for Each Handset Type:")
print(throughput_distribution)


# Task 3.4: Perform K-Means Clustering
X = data[['RTT', 'Throughput']]

# Specify the number of clusters
num_clusters = 3

# Fit K-Means clustering model
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)

# Add cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Display cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)

