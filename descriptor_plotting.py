import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import os

os.makedirs('figures', exist_ok=True)

# df prep
df_sc = pd.read_csv('../02_scPDB/scPDB_descriptors.csv')
df_sc.set_index("protein_code", inplace=True)
df_sc = df_sc.dropna()
df_sc = df_sc.drop(df_sc.columns[:1], axis=1)
df_sc.index = df_sc.index + "_sc"
df_iri = pd.read_csv('../01_Iridium/iridium_desc.csv')
df_iri.set_index("protein_code", inplace=True)
df_iri = df_iri.dropna()
df_iri = df_iri.drop(df_iri.columns[:1], axis=1)
df_iri.index = df_iri.index + "_iri"
df_kel = pd.read_csv('../03_KELCH/kelch_descriptors.csv')
df_kel.set_index("protein_code", inplace=True)
df_kel = df_kel.dropna()
df_kel = df_kel.drop(df_kel.columns[:1], axis=1)
df_kel.index = df_kel.index + "_kel"

# combine df
df = pd.concat([df_sc, df_iri, df_kel], axis=0)
threshold = 0.9
# Calculate the proportion of zeros in each column
zero_proportion = (df == 0).sum() / len(df)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop = zero_proportion[zero_proportion > threshold].index
df = df.drop(columns=columns_to_drop)
# scale
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# do pca
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)

# to check how much variance components explain
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

# Plot the results with different colors for each dataset
plt.figure(figsize=(10, 6))

# Create boolean masks to select rows from each dataset
mask_dataset1 = df.index.str.contains('sc')
mask_dataset2 = df.index.str.contains('iri')
mask_dataset3 = df.index.str.contains('kel')

# Plot Dataset 1 in blue
plt.scatter(principal_components[mask_dataset1, 0], principal_components[mask_dataset1, 1],
            color='cornflowerblue', label='scPDB', s=10)

# Plot Dataset 2 in orange
plt.scatter(principal_components[mask_dataset2, 0], principal_components[mask_dataset2, 1],
            color='orange', label='Iridium', s=10)

# Plot Dataset 3 in red
plt.scatter(principal_components[mask_dataset3, 0], principal_components[mask_dataset3, 1],
            color='red', label='Kelch', s=10)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.savefig('figures/PCA.svg', format='svg', transparent=True)
plt.show()