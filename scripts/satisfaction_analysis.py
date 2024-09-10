import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import psycopg2
from data_cleaning import load_dataset
from db_connection import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


# print(data.columns)
def calculate_scores(data, engagement_clusters, experience_clusters):
    # Calculate engagement and experience scores
    data['Engagement_Score'] = calculate_euclidean_distance(data, engagement_clusters)
    data['Experience_Score'] = calculate_euclidean_distance(data, experience_clusters)
    return data

def calculate_euclidean_distance(data, clusters):
    # Calculate Euclidean distance
    distances = []
    for i in range(len(data)):
        point = data.iloc[i][['RTT', 'Throughput']]  # Assuming 'RTT' and 'Throughput' are columns in your dataset
        cluster_center = clusters.iloc[0]  # Assuming the first row represents cluster center
        distance = ((point - cluster_center) ** 2).sum() ** 0.5
        distances.append(distance)
    return distances

def build_regression_model(data):
    X = data[['Engagement_Score', 'Experience_Score']]
    y = data['Satisfaction_Score']
    regression_model = LinearRegression()
    regression_model.fit(X, y)
    return regression_model

def run_kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    data['Cluster'] = kmeans.fit_predict(data[['Engagement_Score', 'Experience_Score']])
    return data

def export_to_postgres(data, host, port, user, password, database):
    conn = psycopg2.connect(host={DB_HOST}, port={DB_PORT}, user={DB_USER}, password={DB_PASSWORD}, database={DB_NAME})
    cursor = conn.cursor()
    
    # Export data to PostgreSQL
    data.to_sql('satisfaction_scores', conn, if_exists='replace', index=False)
    
    conn.commit()
    conn.close()

def main():
    # Load data and perform previous analyses
    # Load the dataset
    file_path = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\loaded_data.csv"
    file_path1 = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\engagement_clusters.csv"
    file_path2 = r"C:\Users\Tigabu Abriham\Desktop\week2\scripts\experience_clusters.csv"
    
    data = load_dataset(file_path)
    engagement_clusters = load_dataset(file_path1)
    experience_clusters = load_dataset(file_path2)
    
    # Calculate scores
    data = calculate_scores(data, engagement_clusters, experience_clusters)
    
    # Build regression model
    regression_model = build_regression_model(data)
    
    # Run K-Means clustering
    clustered_data = run_kmeans_clustering(data, num_clusters=2)
    
    # Export data to PostgreSQL
    export_to_postgres(clustered_data, host={DB_HOST}, port={DB_PORT}, user={DB_USER}, password={DB_PASSWORD}, database={DB_NAME})

if __name__ == "__main__":
    main()