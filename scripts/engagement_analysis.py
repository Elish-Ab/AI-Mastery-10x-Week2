import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from data_cleaning import load_dataset


file_path = r"C:\Users\Tigabu Abriham\Desktop\week2\notebooks\loaded_data.csv"
    
df = load_dataset(file_path)

df.describe()

df.info()

df.isnull().sum()

df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

df.dropna(inplace=True)

# Fill missing values in numerical columns with the mean
df['Avg RTT DL (ms)'].fillna(df['Avg RTT DL (ms)'].mean(), inplace=True)
df['Avg RTT UL (ms)'].fillna(df['Avg RTT UL (ms)'].mean(), inplace=True)

# Fill missing values in categorical columns with the most frequent value
df['Last Location Name'].fillna(df['Last Location Name'].mode()[0], inplace=True)

# Drop columns with a high number of missing values
threshold = 0.5  # Set the threshold for the percentage of missing values
cols_to_drop = df.columns[df.isnull().mean() > threshold]
df.drop(cols_to_drop, axis=1, inplace=True)

print(df.columns)

# Aggregate engagement metrics per customer ID
engagement_metrics = df.groupby('MSISDN/Number').agg({
    'Bearer Id': 'count',  # Count the number of unique Bearer Ids
    'Dur. (ms)': 'sum',  # Sum of total duration
    'Total UL (Bytes)': 'sum',  # Sum of total upload bytes
    'Total DL (Bytes)': 'sum'  # Sum of total download bytes
})

# Rename the columns for better understanding (optional)
engagement_metrics.columns = ['Num_of_Unique_Bearer_Id', 'Total_Duration', 'Total_UL_Bytes', 'Total_DL_Bytes']

# Report top 10 customers per engagement metric
top_10_sessions_frequency = engagement_metrics['Num_of_Unique_Bearer_Id'].nlargest(10)
top_10_session_duration = engagement_metrics['Total_Duration'].nlargest(10)
top_10_session_total_ul_bytes = engagement_metrics['Total_UL_Bytes'].nlargest(10)
top_10_session_total_dl_bytes = engagement_metrics['Total_DL_Bytes'].nlargest(10)

# Export data to a CSV file
df.to_csv('engagement_clusters.csv', index=False)
print("Top 10 Customers by Number of Unique Bearer Ids:")
print(top_10_sessions_frequency)

print("Top 10 Customers by Total Duration:")
print(top_10_session_duration)

print("Top 10 Customers by Total UL Bytes:")
print(top_10_session_total_ul_bytes)

print("Top 10 Customers by Total DL Bytes:")
print(top_10_session_total_dl_bytes)

# Normalize engagement metrics
scaler = StandardScaler()
normalized_engagement_metrics = scaler.fit_transform(engagement_metrics)

# Perform K-Means clustering with k=3
kmeans = KMeans(n_clusters=3, random_state=0)
engagement_clusters = kmeans.fit_predict(normalized_engagement_metrics)

# Compute and print metrics for each cluster
engagement_metrics['Cluster'] = engagement_clusters
cluster_summary = engagement_metrics.groupby('Cluster').agg(['min', 'max', 'mean', 'sum'])

# Visualize results
# Plotting the mean values of each metric for each cluster
cluster_summary['Num_of_Unique_Bearer_Id']['mean'].plot(kind='bar', color='skyblue', alpha=0.8, legend=True, label='Num_of_Unique_Bearer_Id')
cluster_summary['Total_Duration']['mean'].plot(kind='bar', color='salmon', alpha=0.8, legend=True, label='Total_Duration')
cluster_summary['Total_UL_Bytes']['mean'].plot(kind='bar', color='lightgreen', alpha=0.8, legend=True, label='Total_UL_Bytes')
cluster_summary['Total_DL_Bytes']['mean'].plot(kind='bar', color='orange', alpha=0.8, legend=True, label='Total_DL_Bytes')

plt.xlabel('Cluster')
plt.ylabel('Mean Value')
plt.title('Mean Metrics for Each Cluster')
plt.legend()
plt.show()

# Aggregate user total traffic per application
user_app_traffic = df.groupby('Handset Manufacturer').agg({'Total UL (Bytes)': 'sum', 'Total DL (Bytes)': 'sum'})

# Combine total UL and DL traffic to get total traffic
user_app_traffic['Total Traffic'] = user_app_traffic['Total UL (Bytes)'] + user_app_traffic['Total DL (Bytes)']

# Report top 10 most engaged users per application
top_10_users_per_app = user_app_traffic['Total Traffic'].groupby('Handset Manufacturer').nlargest(10)
print("Top 10 Most Engaged Users per Handset Manufacturer:")
print(top_10_users_per_app)

# Top 3 Application mostly used

# Plot the top 3 most used applications
top_3_applications = user_app_traffic['Total Traffic'].nlargest(3)
top_3_applications.plot(kind='bar', title='Top 3 Most Used Applications')
plt.xlabel('Handset Manufacturer')
plt.ylabel('Total Traffic')
plt.show()

# Determine the optimal value of k using the elbow method
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(normalized_engagement_metrics)
    distortions.append(sum(np.min(cdist(normalized_engagement_metrics, kmeans.cluster_centers_, 'euclidean'), axis=1)) / normalized_engagement_metrics.shape[0])

# Plot the elbow curve
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method for Optimal k')
plt.show()


