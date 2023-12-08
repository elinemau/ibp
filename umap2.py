import pandas as pd
import umap
import matplotlib.pyplot as plt

#df prep
df_sc = pd.read_csv('scPDB_descriptors.csv')
df_sc.set_index("protein_code", inplace=True)
df_sc = df_sc.dropna()
df_sc = df_sc.drop(df_sc.columns[:1], axis=1)
df_sc.index = df_sc.index + "_sc"
df_iri = pd.read_csv('iridium_desc.csv')
df_iri.set_index("protein_code", inplace=True)
print(len(df_iri))
df_iri = df_iri.dropna()
print(len(df_iri))
df_iri = df_iri.drop(df_iri.columns[:1], axis=1)
df_iri.index = df_iri.index + "_iri"

#combine df
df = pd.concat([df_sc, df_iri], axis=0)
threshold = 0.9
# Calculate the proportion of zeros in each column
zero_proportion = (df == 0).sum() / len(df)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop = zero_proportion[zero_proportion > threshold].index
df = df.drop(columns=columns_to_drop)
# Select the relevant features for training UMAP
features = df.columns

# Convert the DataFrame to a NumPy array
data_array = df[features].values
data_array = pd.DataFrame(data_array).dropna().values
# Normalize the data (optional but often recommended)
data_array_normalized = (data_array - data_array.min(axis=0)) / (data_array.max(axis=0) - data_array.min(axis=0))

df['dataset'] = df.index.map(lambda x: 'sc' if 'sc' in x else 'iri')
# Specify the number of dimensions for the UMAP projection
n_components = 2

# Create and fit the UMAP model
umap_model = umap.UMAP(n_components=n_components)
umap_result = umap_model.fit_transform(data_array_normalized)

# Plot the UMAP result
plt.figure(figsize=(10, 6))
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=df['dataset'].map({'sc':"orange", 'iri':'blue'}), s=3)  # Adjust color and size as needed
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('UMAP.svg', format='svg', transparent=True)
plt.show()