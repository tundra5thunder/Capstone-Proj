import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans,DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import webbrowser

# Define the Power BI dashboard URL
power_bi_dashboard_url = "https://app.powerbi.com/groups/me/reports/32fec00c-da80-4deb-a6ff-3705b0210abc/ReportSectionfa87efce70020a487794?experience=power-bi"

def open_power_bi_dashboard():
    # Function to open the Power BI dashboard in the default web browser
    webbrowser.open_new_tab(power_bi_dashboard_url)


def clean_data(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Remove rows with null values
    df = df.dropna()

    # Handle other data defects as needed, such as outliers, incorrect data types, etc.
    # For example, if you have outliers in numerical columns, you can handle them using techniques like winsorization or removing them.

    return df


# Function to load data
def load_data():
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        original_df = df.copy()  # Make a copy of the original dataframe
        return df, original_df
    else:
        return None, None
# Function for data preprocessing
def preprocess_data(df):
    # Label encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = le.fit_transform(df[col])

    # Feature scaling
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled

# Function for feature selection
def select_features(X, y, k):
    selector = SelectKBest(score_func=mutual_info_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected

# Function for dimensionality reduction using PCA
def apply_pca(X, n_components):
    pca = PCA(n_components=n_components,random_state=42)
    X_pca = pca.fit_transform(X)
    return X_pca

# Function for DBSCAN clustering
def dbscan_clustering(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return labels
# Function for K-means clustering
def kmeans_clustering(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)
    labels = kmeans.labels_
    return labels

# Function for cluster analysis
def cluster_analysis(df, cluster_labels):
    st.subheader("Cluster Analysis")

    # Group data by cluster labels
    cluster_groups = df.groupby(cluster_labels)

    # Analyze categorical variables within each cluster
    st.write("Categorical Variables Distribution within Clusters:")
    for cluster_label, cluster_data in cluster_groups:
        st.write(f"Cluster {cluster_label}:")
        for col in cluster_data.select_dtypes(include='object').columns:
            st.write(f"{col}:")
            st.write(cluster_data[col].value_counts())
            st.write("\n")

    # Analyze numerical variables within each cluster
    st.write("Summary Statistics of Numerical Variables within Clusters:")
    for cluster_label, cluster_data in cluster_groups:
        st.write(f"Cluster {cluster_label}:")
        st.write(cluster_data.describe())
        st.write("\n")


# Main function
def main():
    #st.title('K-means Clustering App')

    st.set_page_config(page_title="Clustering App", page_icon=":chart_with_upwards_trend:", layout="wide")
    # Example 2: Adding color to a title with a background color
    st.markdown("<h1 style='color:red;text-align:center;'>Patient Health Stratification</h1>", unsafe_allow_html=True)
    #st.title('Patient Health Stratification')
    #"background-color:#7FBEF7"
    if st.button("Dashboard"):
        open_power_bi_dashboard()
    
    # Load data
    df, df_copy = load_data()

    if df is not None:
        # Clean data
        df = clean_data(df)

        # Sidebar for user input
        k_clusters = st.sidebar.slider('Number of clusters', min_value=2, max_value=5, value=4, step=1)
        k_features = st.sidebar.slider('Number of features to select', min_value=1, max_value=5, value=5, step=1)
        pca_components = st.sidebar.slider('Number of PCA components', min_value=2, max_value=5, value=5, step=1)

        if st.sidebar.button("Run K-means Clustering"):
            # Preprocess data
            X = df.drop(columns=['General_Health', 'Patient_ID'])
            y = df['General_Health']
            X_scaled = preprocess_data(X)

            # Feature selection
            X_selected = select_features(X_scaled, y, k_features)

            # Dimensionality reduction using PCA
            X_pca = apply_pca(X_selected, pca_components)

            # Perform K-means clustering
            cluster_labels = kmeans_clustering(X_pca, k_clusters)

            # Evaluate clustering performance
            silhouette_avg = silhouette_score(X_pca, cluster_labels)
            db_index = davies_bouldin_score(X_pca, cluster_labels)
            ch_index = calinski_harabasz_score(X_pca, cluster_labels)

            df_copy['Predicted_labels'] = cluster_labels
            df_copy = df_copy.drop('General_Health', axis=1)

            # Display results
            st.subheader("Clustering Results")
            st.write("Clustering Result with Labels")
            st.write(df_copy)
            st.write("Silhouette Score:", silhouette_avg)
            st.write("Davies–Bouldin index:", db_index)
            st.write("Calinski-Harabasz index:", ch_index)

            # Cluster analysis
            cluster_analysis(df, cluster_labels)

            # Plot clustered data
            st.subheader("Clustered Data Visualization")
            df_clustered = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'PC3': X_pca[:, 2], 'Cluster': cluster_labels})
            fig = px.scatter_3d(df_clustered, x='PC1', y='PC2', z='PC3', color='Cluster', title='3D Scatter Plot with K-means Clustering')
            st.plotly_chart(fig)
            
           # df_copy.to_csv(r'C:\Users\shubham.raj.lv\Documents\predicted_values_2.csv', index=False)
            
if __name__ == '__main__':
    main()





