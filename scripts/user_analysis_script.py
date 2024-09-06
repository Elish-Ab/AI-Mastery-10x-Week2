import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from db_connection import create_connection

# Function to extract data from PostgreSQL
def extract_data_from_postgres(query):
    try:
        # Establish a connection to the database
        conn = create_connection()
        
        # Load data using pandas
        df = pd.read_sql_query(query, conn)
        
        # Close the database connection
        conn.close()
        
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Query to extract data from xdr_data table
query = "SELECT * FROM xdr_data;"

# Call function to extract data
df = extract_data_from_postgres(query)
# print(df.columns)
# Task 1.1 - Aggregate user behavior data
def aggregate_user_behavior_data(data):
    # Convert column names to lowercase for case-insensitive matching
    data.columns = data.columns.str.lower()
    
    # Define the columns to aggregate by
    aggregation_cols = ['youtube dl (bytes)', 'youtube ul (bytes)', 'netflix dl (bytes)', 
                        'netflix ul (bytes)', 'gaming dl (bytes)', 'gaming ul (bytes)', 
                        'other dl (bytes)', 'other ul (bytes)', 'total ul (bytes)', 'total ul (bytes)']
    
    # Group by 'MSISDN/Number' and aggregate by sum for the specified columns
    user_data = data.groupby('msisdn/number')[aggregation_cols].sum()
    
    return user_data

# Task 1.2 - Exploratory Data Analysis
def describe_data(data):
    data_description = data.describe()

def handle_missing_values_and_outliers(data):
    data.fillna(data.mean(), inplace=True)
    # Handle outliers using z-score or any other method


def clean_column_names(data):
    # Remove leading and trailing whitespaces from column names
    data.columns = data.columns.str.strip()
    return data

def segment_users(data):
    # Clean the column names
    data = clean_column_names(data)
    print(data.columns)
    # Check if the column exists after cleaning
    if 'youtube dl (bytes)' in df.columns:
        # Perform the segmentation
        data['Duration_Decile'] = pd.qcut(data['youtube dl (bytes)'], q=5, labels=False)
    else:
        print("Column 'youtube dl (bytes)' not found after cleaning column names.")
        return data  # Return the DataFrame without further processing if the column is missing
    
    return data

def basic_metrics_analysis(data):
    # Check if the columns exist after cleaning
    columns_to_agg = [col for col in data.columns if 'youtube dl (bytes)' in col or 'total ul' in col or 'netflix dl' in col]

    if columns_to_agg:
        basic_metrics = data[columns_to_agg].agg(['mean', 'median'])
        print(basic_metrics)
    else:
        print("Columns for basic metrics analysis not found in the DataFrame.")

def univariate_analysis(data):
    # Get the positions of columns based on substrings
    dur_col = data.columns.get_loc([col for col in data.columns if 'Duration_Decile' in col][0])
    ul_col = data.columns.get_loc([col for col in data.columns if 'youtube ul (bytes)' in col][0])
    dl_col = data.columns.get_loc([col for col in data.columns if 'youtube dl (bytes)' in col][0])

# Check if the columns exist after cleaning
    if dur_col and ul_col and dl_col:
        dispersion_parameters = data.iloc[:, [dur_col, ul_col, dl_col]].describe()
        print(dispersion_parameters)
    else:
        print("Columns for univariate analysis not found in the DataFrame.")

def graphical_univariate_analysis(data):
    # Check if the columns exist in the DataFrame
    columns_to_plot = ['youtube dl (bytes)', 'total ul (bytes)', 'netflix dl (bytes)']
    missing_columns = [col for col in columns_to_plot if col not in data.columns]

    if missing_columns:
        for col in missing_columns:
            print(f"Column '{col}' not found in the DataFrame.")
    else:
        plt.figure(figsize=(18, 6))

        for i, column in enumerate(columns_to_plot, 1):
            plt.subplot(1, 3, i)
            sns.histplot(data[column], kde=True)
            plt.title(f'{column} Distribution')

        plt.tight_layout()
        plt.show()

# def bivariate_analysis(data):
#     # Columns to plot
#     columns_to_plot = ['Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)',
#                        'YouTube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)',
#                        'Other UL (Bytes)', 'Social Media DL (Bytes)', 'Google DL (Bytes)',
#                        'Email DL (Bytes)', 'YouTube DL (Bytes)', 'Netflix DL (Bytes)',
#                        'Gaming DL (Bytes)', 'Other DL (Bytes)', 'Total UL (Bytes)',
#                        'total ul (bytes)']

#     missing_columns = [col for col in columns_to_plot if col not in data.columns]

#     if missing_columns:
#         for col in missing_columns:
#             print(f"Column '{col}' not found in the DataFrame.")
#     else:
#         sns.pairplot(data[columns_to_plot])

def correlation_analysis(data):
    # Columns for correlation analysis
    columns_for_correlation = ['youtube dl (bytes)', 'youtube ul (bytes)', 'netflix dl (bytes)',
                                'netflix ul (bytes)', 'gaming dl (bytes)', 'gaming ul (bytes)',
                                'other dl (bytes)', 'other ul (bytes)', 'total ul (bytes)',
                                'total ul (bytes)', 'Duration_Decile']
    print(data.columns)
    missing_columns = [col for col in columns_for_correlation if col not in data.columns]

    if missing_columns:
        for col in missing_columns:
    
            print(f"missing Column '{col}' not found in the DataFrame.")
    else:
        correlation_matrix = data[columns_for_correlation].corr()
        sns.heatmap(correlation_matrix, annot=True)

def dimensionality_reduction(data):
    columns_for_reduction = ['Social Media UL (Bytes)', 'Google UL (Bytes)', 'Email UL (Bytes)',
                             'YouTube UL (Bytes)', 'Netflix UL (Bytes)', 'Gaming UL (Bytes)',
                             'Other UL (Bytes)', 'Social Media DL (Bytes)', 'Google DL (Bytes)',
                             'Email DL (Bytes)', 'YouTube DL (Bytes)', 'Netflix DL (Bytes)',
                             'Gaming DL (Bytes)', 'Other DL (Bytes)']
    print(df.columns)
    missing_columns = [col for col in columns_for_reduction if col not in data.columns]

    if missing_columns:
        for col in missing_columns:
            print(f" dimenColumn '{col}' not found in the DataFrame.")
        return None

    features = data[columns_for_reduction]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)
    principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    
    return principal_df

def main():
    user_data = aggregate_user_behavior_data(df)
    describe_data(user_data)
    handle_missing_values_and_outliers(user_data)
    segment_users(user_data)
    basic_metrics_analysis(user_data)
    univariate_analysis(user_data)
    graphical_univariate_analysis(user_data)
    # bivariate_analysis(user_data)
    correlation_analysis(user_data)
    pca_results = dimensionality_reduction(user_data)

if __name__ == "__main__":
    main()