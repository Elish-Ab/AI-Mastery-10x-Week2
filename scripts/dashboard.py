import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from data_cleaning import load_dataset
# Importing user overview analysis
from user_analysis_script import graphical_univariate_analysis, bivariate_analysis, correlation_analysis

# Importing engagement analysis functions
from engagement_analysis import df, engagement_metrics, scaler, normalized_engagement_metrics, cluster_summary, user_app_traffic, top_10_users_per_app, top_3_applications, K, distortions

# Importing experience analysis 
from experience_analysis import perform_experience_analysis

# Importing satisfaction data
from satisfaction_analysis import calculate_scores, run_kmeans_clustering


# Load your data for all tasks here
file_path = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\loaded_data.csv"   
data = load_dataset(file_path)

# Task 1: User Overview Analysis
def user_overview_analysis():
    st.title('User Overview Analysis')
    
        # Graphical Univariate Analysis
    st.subheader('Graphical Univariate Analysis')
    st.write("Visualizing the distribution of selected columns")
    graphical_univariate_analysis(df)  

    # Bivariate Analysis
    st.subheader('Bivariate Analysis')
    st.write("Pairplot of selected columns")
    bivariate_plot = bivariate_analysis(df)
    st.pyplot(bivariate_plot)

    # Correlation Analysis
    st.subheader('Correlation Analysis')
    st.write("Heatmap of selected columns")
    correlation_plot = correlation_analysis(df)
    st.pyplot(correlation_plot)

# Task 2: User Engagement Analysis
def user_engagement_analysis():
    st.title('User Engagement Analysis Visualization')

    # Plotting the mean values of each metric for each cluster
    st.subheader('Mean Metrics for Each Cluster')
    fig, ax = plt.subplots()
    cluster_summary['Num_of_Unique_Bearer_Id']['mean'].plot(kind='bar', color='skyblue', alpha=0.8, legend=True, label='Num_of_Unique_Bearer_Id', ax=ax)
    cluster_summary['Total_Duration']['mean'].plot(kind='bar', color='salmon', alpha=0.8, legend=True, label='Total_Duration', ax=ax)
    cluster_summary['Total_UL_Bytes']['mean'].plot(kind='bar', color='lightgreen', alpha=0.8, legend=True, label='Total_UL_Bytes', ax=ax)
    cluster_summary['Total_DL_Bytes']['mean'].plot(kind='bar', color='orange', alpha=0.8, legend=True, label='Total_DL_Bytes', ax=ax)

    plt.xlabel('Cluster')
    plt.ylabel('Mean Value')
    plt.title('Mean Metrics for Each Cluster')
    st.pyplot(fig)

    st.subheader('Top 10 Most Engaged Users per Handset Manufacturer')
    st.write(top_10_users_per_app)

    st.subheader('Top 3 Most Used Applications')
    st.bar_chart(top_3_applications)

    st.subheader('Elbow Method for Optimal k')
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Elbow Method for Optimal k')
    st.pyplot()

# Task 3: Experience Analysis
def experience_analysis(file_path):
    st.title('Telecommunication Industry User Experience Analysis Dashboard')
    
    # Perform Experience Analysis
    data, _, _, _, cluster_centers = perform_experience_analysis(file_path)

    # Display top and bottom values
    st.subheader('Top and Bottom Values:')
    st.write('Top Values:')
    st.write(data.nlargest(5, ['TCP', 'RTT', 'Throughput']))
    st.write('Bottom Values:')
    st.write(data.nsmallest(5, ['TCP', 'RTT', 'Throughput']))

    # Data Distribution by Handset Type
    st.subheader('Mean Throughput for Each Handset Type:')
    throughput_distribution = data.groupby('Handset Type')['Throughput'].mean()
    st.write(throughput_distribution)

    # Visualizing the clusters
    st.subheader('K-Means Clustering Visualization:')
    plt.figure(figsize=(8, 6))
    plt.scatter(data['RTT'], data['Throughput'], c=data['Cluster'], cmap='viridis', s=50)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', label='Centroids')
    plt.xlabel('RTT')
    plt.ylabel('Throughput')
    plt.title('K-Means Clustering')
    plt.legend()
    st.pyplot(plt)

    # Export the data
    data.to_csv('experience_clusters.csv', index=False)

# Task 4 Analysis Visualization
def satisfaction_analysis(data):
    st.title('Task 4 Analysis Visualization')
    file_path1 = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\engagement_clusters.csv"
    file_path2 = r"C:\Users\Tigabu Abriham\Desktop\week2\scripts\experience_clusters.csv"
    # Calculate scores
    engagement_clusters = load_dataset(file_path1)  # Load engagement clusters data
    experience_clusters = load_dataset(file_path2)  # Load experience clusters data
    data = calculate_scores(data, engagement_clusters, experience_clusters)

    # Build regression model
    regression_model = build_regression_model(data)

    # Run K-Means clustering
    clustered_data = run_kmeans_clustering(data, num_clusters=2)

    # Visualize Scores and Clustering
    st.subheader('Satisfaction Scores and Clustering')
    st.write(data)

    # Scatter plot of Engagement and Experience Scores
    st.subheader('Scatter plot of Engagement and Experience Scores')
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Engagement_Score'], data['Experience_Score'], c=data['Cluster'], cmap='viridis', s=50)
    plt.xlabel('Engagement Score')
    plt.ylabel('Experience Score')
    plt.title('Engagement vs Experience Scores')
    st.pyplot(plt)

    # Scatter plot with Satisfaction Score
    st.subheader('Scatter plot with Satisfaction Score')
    plt.figure(figsize=(8, 6))
    plt.scatter(data['Engagement_Score'], data['Experience_Score'], c=data['Satisfaction_Score'], cmap='viridis', s=50)
    plt.colorbar()
    plt.xlabel('Engagement Score')
    plt.ylabel('Experience Score')
    plt.title('Engagement vs Experience Scores with Satisfaction Score')
    st.pyplot(plt)


# Main Streamlit App
def main():
    st.sidebar.title('Dashboard Navigation')
    page = st.sidebar.radio('Go to', ['User Overview Analysis', 'User Engagement Analysis', 'Experience Analysis', 'Satisfaction Analysis'])

    if page == 'User Overview Analysis':
        user_overview_analysis()
    elif page == 'User Engagement Analysis':
        user_engagement_analysis()
    elif page == 'Experience Analysis':
        file_path = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\loaded_data.csv"
        experience_analysis(file_path)
    elif page == 'Satisfaction Analysis':
        file_path = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\loaded_data.csv"
        data = load_dataset(file_path)
        satisfaction_analysis(data)

if __name__ == '__main__':
    main()