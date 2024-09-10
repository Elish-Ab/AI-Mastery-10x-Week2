import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from data_cleaning import load_dataset, handle_missing_values, handle_outliers

def perform_experience_analysis(file_path):
    # Load the dataset
    data = load_dataset(file_path)
    data = handle_missing_values(data)
    data = handle_outliers(data)

    # Aggregate Customer Information
    agg_data = data.groupby('Bearer Id').agg({
        'TCP': 'mean',
        'RTT': 'mean',
        'Throughput': 'mean',
        'Handset Type': lambda x: x.mode().iloc[0] if not x.mode().empty else None
    }).reset_index().rename(columns={'Handset Type': 'Mode_Handset_Type'})

    # List Top and Bottom Values
    top_values = data.nlargest(5, ['TCP', 'RTT', 'Throughput'])
    bottom_values = data.nsmallest(5, ['TCP', 'RTT', 'Throughput'])

    # Analyze Data Distribution
    throughput_distribution = data.groupby('Handset Type')['Throughput'].mean()

    # Perform K-Means Clustering
    X = data[['RTT', 'Throughput']]
    imp = SimpleImputer(strategy='mean')
    X_imputed = imp.fit_transform(X)
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X_imputed)
    data['Cluster'] = kmeans.labels_
    
    #visulaization
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

    # Export the data
    data.to_csv('experience_clusters.csv', index=False)
    


# Example usage
file_path = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\loaded_data.csv"
