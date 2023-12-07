import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns


df = pd.read_csv('scPDB_descriptors.csv')
df.set_index("protein_code", inplace=True)
df = df.dropna()
df = df.drop(df.columns[:1], axis=1)
threshold = 0.9
# Calculate the proportion of zeros in each column
zero_proportion = (df == 0).sum() / len(df)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop = zero_proportion[zero_proportion > threshold].index
df = df.drop(columns=columns_to_drop)
print(df.head())
#standardize the data
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df)

#do pca
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

#to check how much variance components explain
"""explained_variance_ratio = pca.explained_variance_ratio_
# Plot the cumulative explained variance
cumulative_explained_variance = explained_variance_ratio.cumsum()
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance vs. Number of Principal Components')
plt.show()"""

# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

"""# Plot the 2D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(pc_df['PC1'], pc_df['PC2'])
plt.title('2D Scatter Plot with PCA')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.grid(True)
plt.show()"""

# Plot the data in 3D
"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pc_df['PC1'], pc_df['PC2'], pc_df['PC3'])
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('Scatter Plot of Data in 3D with First 3 PCs')
plt.show()"""

#clustering
kmeans = KMeans(n_clusters=10)
pc_df['Cluster'] = kmeans.fit_predict(principal_components)
#cluster_centroids = kmeans.cluster_centers_
# Plot the data in 3D with cluster centroids
# Plot cluster centroids
plt.scatter(pc_df['PC1'], pc_df['PC2'],
           c=pc_df['Cluster'], cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Scatter Plot of Data with Cluster Centroids in 3D')
plt.colorbar(label='Cluster')
plt.show()

