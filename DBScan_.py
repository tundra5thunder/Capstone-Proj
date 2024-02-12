import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


df=pd.read_csv("C:\\Users\\sandhya.g.lv\\Downloads\\New.csv")
#df.columns

#Seperating numerical and categorical columns
def sep_col():
   num_col=df.columns[df.dtypes != "object"]
   print("Numerical Columns: ",num_col)
   cat_col=df.columns[df.dtypes=="object"]
   print("Categorical Columns: ",cat_col)
   return cat_col


#Encoding data to convert categorical columns
label_encoder=preprocessing.LabelEncoder()
def enco(df,col):
 for col in sep_col():
  df[col] = label_encoder.fit_transform(df[col])
 return df
df=enco(df,sep_col())


#FEATURE SCALING
from sklearn.preprocessing import StandardScaler



# Assuming 'data' is your dataset with the listed features
# Extract the features to be scaled

def feature_sel():
   features_to_scale = ['General_Health', 'Checkup', 'Exercise', 'Heart_Disease',
                     'Skin_Cancer', 'Other_Cancer', 'Depression', 'Diabetes',
                     'Arthritis', 'Age', 'Height_(cm)', 'Weight_(kg)', 'BMI',
                     'Smoking_History', 'Alcohol_Consumption', 'Fruit_Consumption',
                     'Green_Vegetables_Consumption', 'FriedPotato_Consumption']
   # Extract the feature values from the data
   X = df[features_to_scale]
   # Initialize the scaler
   scaler = StandardScaler()
   # Fit the scaler on the data to compute mean and standard deviation
   scaler.fit(X)
   #Transform the data using the scaler
   X_scaled = scaler.transform(X)
   # Replace the original features with the scaled features in the dataset
   df[features_to_scale] = X_scaled

from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer


# Min-Max Scaling
def min_max_scal():
   scaler_minmax = MinMaxScaler()
   X_minmax_scaled = scaler_minmax.fit_transform(X)
   return X_minmax_scaled


#FEATURE SELECTION
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, annot_kws={"fontsize":10},fmt=".2f")
plt.show()

#PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(min_max_scal())


#DBSCAN
selected_features = [ 'Exercise', 'Diabetes', 'Arthritis', 'BMI', 'Alcohol_Consumption']

# Extract selected features
X = df[selected_features]

# Handle categorical variables (e.g., one-hot encoding for 'Sex')

# Handle missing values if needed

# Scale numerical features
scaler_minmax = MinMaxScaler()
X_minmax_scaled = scaler_minmax.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_minmax_scaled)

# Apply DBSCAN
eps = 0.5 # Adjust epsilon (neighborhood distance) based on your data
min_samples = 5  # Adjust min_samples based on your data

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(X_pca)

# Visualize the clusters
unique_labels = set(cluster_labels)
colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))  # Generate colors for each cluster

plt.figure(figsize=(8, 6))
for label, color in zip(unique_labels, colors):
    if label == -1:
        # Outliers are represented as black points
        cluster_data = X_pca[cluster_labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c='black', label='Outliers')
    else:
        # Non-outlier clusters
        cluster_data = X_pca[cluster_labels == label]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], c='red', label=f'Cluster {label}')

plt.title('DBSCAN Clustering after PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Calculate silhouette score
silhouette_avg = silhouette_score(X_pca, cluster_labels)
print(f"Silhouette Score after PCA and DBSCAN: {silhouette_avg}")


# Import other libraries as needed

def main():
    st.title("Patient stratification")

    # Load data
    df = pd.read_csv("C:\\Users\\sandhya.g.lv\\Downloads\\New.csv")

    # Display dataset
    st.subheader("Dataset")
    st.write(df)

    # Your data preprocessing and analysis code here
    # This could include PCA, feature selection, clustering, etc.
    
    #call seperation of columns
    sep_col()
    

    #Calling label encoding:
    df=enco(df,sep_col())

    #feature selection:
    feature_sel()

    #Minmax scal:
    min_max_scal()


    # Example:
    # Perform PCA
    pca = PCA()
    pca_data = pca.fit_transform(df)

    # Display explained variance ratio
    st.subheader("Explained Variance Ratio")
    st.line_chart(np.cumsum(pca.explained_variance_ratio_))

    # Example clustering using KMeans
    kmeans = KMeans(n_clusters=4, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_data)

    # Display clustering results
    st.subheader("Cluster Labels")
    st.write(cluster_labels)

    # Display scatter plot of clusters
    # You can use Matplotlib or any other plotting library
    # Example:
    fig, ax = plt.subplots()
    ax.scatter(pca_data[:, 0], pca_data[:, 1], c=cluster_labels)
    st.subheader("Scatter Plot of Clusters")
    st.pyplot(fig)

    # Evaluation metrics
    silhouette_avg = silhouette_score(pca_data, cluster_labels)
    db_index = davies_bouldin_score(pca_data, cluster_labels)
    ch_index = calinski_harabasz_score(pca_data, cluster_labels)

    st.subheader("Evaluation Metrics")
    st.write(f"Silhouette Score: {silhouette_avg}")
    st.write(f"Davies–Bouldin index: {db_index}")
    st.write(f"Calinski-Harabasz index: {ch_index}")

if __name__ == "__main__":
    main()

