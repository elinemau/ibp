import pandas as pd
import re
import numpy as np
from sklearn.decomposition import PCA

#making dataframe to make x y z coordinates accessible
df = pd.DataFrame(columns=["number", "atom", "x", "y", "z"])
with open("1a28\\volsite\\CAVITY_N1_ALL.mol2", "r") as file:
    line = file.readline()
    while not line.startswith("@<TRIPOS>ATOM"):
        line = file.readline()
    line = file.readline()
    while not line.startswith("@<TRIPOS>BOND"):
        df.loc[len(df)] = list(filter(lambda x: len(x) > 0, re.split(r'[\n\t\s]+', line)))[0:5]
        line = file.readline()
file.close()
# do check if not empty

df['x'] = df['x'].astype(float)
df['y'] = df['y'].astype(float)
df['z'] = df['z'].astype(float)

point_cloud = df[["x", "y", "z"]].to_numpy()

# Create a PCA instance and specify the number of components you want to retain
pca = PCA(n_components=3)

# Fit the PCA model to your data
pca.fit(point_cloud)

# Get the principal components
principal_components = pca.components_
explained_variance = pca.explained_variance_
print(principal_components, explained_variance)
