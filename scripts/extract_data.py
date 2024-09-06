import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# Check if data extraction was successful
if df is not None:
    print("Data extraction successful. DataFrame shape:", df.shape)
    
    # Task 1.1 - Aggregate per user behavior information
    user_behavior = df.groupby('MSISDN').agg({
        'xDR_sessions': 'sum',
        'Session_Duration': 'sum',
        'Total_DL_UL': 'sum',
        'Social Media DL': 'sum',
        'Social Media UL': 'sum',
        'Google DL': 'sum',
        'Google UL': 'sum',
        'Email DL': 'sum',
        'Email UL': 'sum',
        'YouTube DL': 'sum',
        'YouTube UL': 'sum',
        'Netflix DL': 'sum',
        'Netflix UL': 'sum',
        'Gaming DL': 'sum',
        'Gaming UL': 'sum',
        'Other DL': 'sum',
        'Other UL': 'sum'
    })
    
    # Task 1.2 - Exploratory Data Analysis
    # Data Cleaning: Replace missing values with the mean
    user_behavior.fillna(user_behavior.mean(), inplace=True)
    
    # Variable transformations
    user_behavior['Total_Session_Duration'] = user_behavior['Session_Duration']
    user_behavior['Total_DL_UL'] = user_behavior['Total_DL_UL']
    
    # Basic metrics
    basic_metrics = user_behavior.describe()
    
    # Non-Graphical Univariate Analysis
    dispersion_parameters = user_behavior.quantile([0.25, 0.5, 0.75])
    
    # Graphical Univariate Analysis
    import matplotlib.pyplot as plt
    user_behavior.hist(figsize=(15, 10))
    plt.show()
    
    # Bivariate Analysis
    bivariate_analysis = user_behavior[['Social Media DL', 'Google DL', 'Email DL', 
                                        'YouTube DL', 'Netflix DL', 'Gaming DL', 
                                        'Other DL']].corrwith(user_behavior['Total_DL_UL'])
    
    # Correlation Analysis
    correlation_matrix = user_behavior[['Social Media DL', 'Google DL', 'Email DL', 
                                        'YouTube DL', 'Netflix DL', 'Gaming DL', 
                                        'Other DL']].corr()
    
    # Dimensionality Reduction - PCA
    features = ['Social Media DL', 'Google DL', 'Email DL', 'YouTube DL', 
                'Netflix DL', 'Gaming DL', 'Other DL']
    
    # Standardize the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(user_behavior[features])
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_scaled)
    
    # Print PCA explained variance ratio
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)
    
else:
    print("Failed to extract data.")