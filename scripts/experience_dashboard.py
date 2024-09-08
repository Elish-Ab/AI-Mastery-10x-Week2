import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_cleaning import load_dataset, handle_missing_values, handle_outliers

# Load the dataset
file_path = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\loaded_data.csv"
data = load_dataset(file_path)

# Data Cleaning
data = handle_missing_values(data)
data = handle_outliers(data)

# Perform K-Means Clustering
X = data[['RTT', 'Throughput']]
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
data['Cluster'] = kmeans.labels_

# Streamlit Dashboard
st.title('Telecommunication Industry User Experience Analysis Dashboard')

# Display top and bottom values
st.subheader('Top and Bottom Values:')
st.write('Top Values:')
st.write(data.nlargest(5, ['TCP', 'RTT', 'Throughput']))
st.write('Bottom Values:')
st.write(data.nsmallest(5, ['TCP', 'RTT', 'Throughput']))

# Data Distribution by Handset Type
st.subheader('Mean Throughput for Each Handset Type:')
throughput_distribution = data.groupby('Handset_Type')['Throughput'].mean()
st.write(throughput_distribution)

# Visualizing the clusters
st.subheader('K-Means Clustering Visualization:')
plt.figure(figsize=(8, 6))
plt.scatter(data['RTT'], data['Throughput'], c=data['Cluster'], cmap='viridis', s=50)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids')
plt.xlabel('RTT')
plt.ylabel('Throughput')
plt.title('K-Means Clustering')
plt.legend()
st.pyplot(plt)