import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_excel("C:/Users/32496/OneDrive - KU Leuven/kuleuven(1)/IBP/dataframe.xlsx")
df.set_index("protein_code", inplace=True)
df = df.dropna()

#standardize the data
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df)

#do pca
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(scaled_data)

# Choose the number of clusters (you may need to choose this based on your data or use techniques like the elbow method)
n_clusters = 10

# Apply K-Means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_assignments = kmeans.fit_predict(reduced_data)

# Visualize the clusters
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=cluster_assignments, cmap='viridis')
plt.title('K-Means Clustering (2D PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()