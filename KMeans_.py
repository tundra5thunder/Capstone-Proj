import numpy as np
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif,f_classif
from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer,StandardScaler
from sklearn.decomposition import PCA
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score




df=pd.read_csv("C:\\Users\\sandhya.g.lv\\Downloads\\New.csv")

#SEGREGATION OF NUMERICAL AND CATEGORICAL COLUMN

num_col=df.columns[df.dtypes != "object"]
print("Numerical Columns: ",num_col)
cat_col=df.columns[df.dtypes=="object"]
print("Categorical Columns: ",cat_col)


le=LabelEncoder()
for i in df.columns:
  if(df[i].dtypes=='object'):
    df[i]=le.fit_transform(df[i])

#Perform PCA to determine the number of components to retain
pca = PCA()
pca_data=pca.fit(df)

#Plot explained variance ratio

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance Ratio by Number of Components')
plt.grid(True)
plt.show()

#Define the number of features to select
#Step 1: Scaling the selected features.
k = 5



# Instantiate the SelectKBest object with the desired scoring function
selector = SelectKBest(score_func=mutual_info_classif, k=k)

# Fit the selector to your data, excluding any non-feature columns like target or ID columns
X_selected = selector.fit_transform(df.drop(['General_Health', 'Patient_ID'], axis=1), df['General_Health'])

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Get the names of the selected features
selected_feature_names = df.drop(['General_Health', 'Patient_ID'], axis=1).columns[selected_feature_indices]

# Print the selected feature names
print("Selected Features:")
print(selected_feature_names)


#Scaling


# Min-Max Scaling
scaler_minmax = MinMaxScaler()
X_minmax_scaled = scaler_minmax.fit_transform(X_selected)


#Step 2: Elbow Method for Choosing Number of Clusters
wcss = []  # Within-cluster sum of squares
for i in range(1, 11):  # Test different number of clusters from 1 to 10
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_minmax_scaled)
    wcss.append(kmeans.inertia_)

#Plot the Elbow Curve
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

#Step 3: Clustering (K-Means)

kmeans = KMeans(n_clusters=4, random_state=42)  # Example: Cluster into 4 groups
kmeans.fit(X_minmax_scaled)

#Step 4: Scatter Plot of Different Health Categories

plt.figure(figsize=(10, 6))
for cluster_label in np.unique(kmeans.labels_):
    plt.scatter(X_minmax_scaled[kmeans.labels_ == cluster_label, 0],X_minmax_scaled[kmeans.labels_ == cluster_label, 1], label=f'Cluster {cluster_label}')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of Different Health Categories')
plt.legend()
plt.show()

#Step 5: Scatter Plot of Different Health Categories with 3D Visualization

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for cluster_label in range(4):
    cluster_points = X_minmax_scaled[kmeans.labels_ == cluster_label]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {cluster_label}')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D Scatter Plot of Different Health Categories')
ax.legend()
plt.show()


#Step 6: Using Metrics to test the accuracy and selecting the model.

# Compute the silhouette score
silhouette_avg = silhouette_score(X_minmax_scaled, kmeans.labels_)

print(f"Silhouette Score: {silhouette_avg}")



# Assuming you have already fitted your k-means model and obtained the cluster labels
# kmeans.fit(X)

# Calculate the Davies–Bouldin index
db_index = davies_bouldin_score(X_minmax_scaled, kmeans.labels_)
print(f"Davies–Bouldin index: {db_index}")

# Assuming you have already fitted your k-means model and obtained the cluster labels
# kmeans.fit(X)

# Calculate the Calinski-Harabasz index
ch_index = calinski_harabasz_score(X_minmax_scaled, kmeans.labels_)
print(f"Calinski-Harabasz index: {ch_index}")




# Fit K-means clustering algorithm
kmeans = KMeans(n_clusters=4, random_state=42)
cluster_labels = kmeans.fit_predict(X_minmax_scaled)
# Create a DataFrame with the principal components and cluster labels
df_clusters = pd.DataFrame({'PC1': X_minmax_scaled[:, 0], 'PC2': X_minmax_scaled[:, 1], 'PC3': X_minmax_scaled[:, 2], 'Cluster': cluster_labels})

# Plot 3D scatter plot with cluster labels
fig = px.scatter_3d(df_clusters, x='PC1', y='PC2', z='PC3', color='Cluster',
                    title='Clusters from K-means Algorithm (3D Scatter Plot)',
                    labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2', 'PC3': 'Principal Component 3'},
                    opacity=0.8, symbol='Cluster')

# Show the plot
fig.show()

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
