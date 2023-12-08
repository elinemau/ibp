import pandas as pd
import umap.umap_ as umap
import matplotlib.pyplot as plt
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
df_volsite = df.iloc[:, 1:89]
df_algorithm = df.iloc[:, 90:140]

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

df['dataset'] = df.index.map(lambda x: 'sc' if 'sc' in x else ('iri' if 'iri' in x else 'kel'))
# Specify the number of dimensions for the UMAP projection
n_components = 2

# Create and fit the UMAP model
umap_model = umap.UMAP(n_components=n_components)
umap_result = umap_model.fit_transform(data_array_normalized)

# Plot the UMAP result
plt.figure(figsize=(10, 6))
plt.scatter(umap_result[:, 0], umap_result[:, 1], c=df['dataset'].map({'sc': "orange", 'iri': 'blue', 'kel': 'red'}),
            s=3)  # Adjust color and size as needed
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('figures/UMAP.svg', format='svg', transparent=True)
plt.show()


# VOLSITE

zero_proportion_volsite = (df_volsite == 0).sum() / len(df_volsite)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop_volsite = zero_proportion_volsite[zero_proportion_volsite > threshold].index
df_volsite = df_volsite.drop(columns=columns_to_drop_volsite)
# Select the relevant features for training UMAP
features_volsite = df_volsite.columns

# Convert the DataFrame to a NumPy array
data_array_volsite = df_volsite[features_volsite].values
data_array_volsite = pd.DataFrame(data_array_volsite).dropna().values
# Normalize the data (optional but often recommended)
data_array_volsite_normalized = (data_array_volsite - data_array_volsite.min(axis=0)) / (
            data_array_volsite.max(axis=0) - data_array_volsite.min(axis=0))

df_volsite['dataset'] = df_volsite.index.map(lambda x: 'sc' if 'sc' in x else ('iri' if 'iri' in x else 'kel'))
# Specify the number of dimensions for the UMAP projection
n_components = 2

# Create and fit the UMAP model
umap_model_volsite = umap.UMAP(n_components=n_components)
umap_result_volsite = umap_model_volsite.fit_transform(data_array_volsite_normalized)

# Plot the UMAP result
plt.figure(figsize=(10, 6))
plt.scatter(umap_result_volsite[:, 0], umap_result_volsite[:, 1],
            c=df_volsite['dataset'].map({'sc': "orange", 'iri': 'blue', 'kel': 'red'}),
            s=3)  # Adjust color and size as needed
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('figures/UMAP_volsite.svg', format='svg', transparent=True)
plt.show()

# ALGORITHM

zero_proportion_alg = (df_algorithm == 0).sum() / len(df_algorithm)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop_alg = zero_proportion_alg[zero_proportion_alg > threshold].index
df_algorithm = df_algorithm.drop(columns=columns_to_drop_alg)
# Select the relevant features for training UMAP
features_alg = df_algorithm.columns

# Convert the DataFrame to a NumPy array
data_array_alg = df_algorithm[features_alg].values
data_array_alg = pd.DataFrame(data_array_alg).dropna().values
# Normalize the data (optional but often recommended)
data_array_alg_normalized = (data_array_alg - data_array_alg.min(axis=0)) / (
            data_array_alg.max(axis=0) - data_array_alg.min(axis=0))

df_algorithm['dataset'] = df_algorithm.index.map(lambda x: 'sc' if 'sc' in x else ('iri' if 'iri' in x else 'kel'))
# Specify the number of dimensions for the UMAP projection
n_components = 2

# Create and fit the UMAP model
umap_model_alg = umap.UMAP(n_components=n_components)
umap_result_alg = umap_model_alg.fit_transform(data_array_alg_normalized)

# Plot the UMAP result
plt.figure(figsize=(10, 6))
plt.scatter(umap_result_alg[:, 0], umap_result_alg[:, 1],
            c=df_algorithm['dataset'].map({'sc': "orange", 'iri': 'blue', 'kel': 'red'}),
            s=3)  # Adjust color and size as needed
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.savefig('figures/UMAP_algorithm.svg', format='svg', transparent=True)
plt.show()
