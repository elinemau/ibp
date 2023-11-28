import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_excel("C:/Users/32496/OneDrive - KU Leuven/kuleuven(1)/IBP/dataframe.xlsx")
df.set_index("protein_code", inplace=True)
df = df.dropna()

#standardize the data
scaler=StandardScaler()
scaled_data=scaler.fit_transform(df)

#do pca
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)


# Create a DataFrame with the principal components
pc_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Plot the 2D scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(pc_df['PC1'], pc_df['PC2'])
plt.title('2D Scatter Plot with PCA')
plt.xlabel('Principal Component 1 (PC1)')
plt.ylabel('Principal Component 2 (PC2)')
plt.grid(True)
plt.show()
