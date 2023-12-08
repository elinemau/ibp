import csv
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv("full_alg_iridium_scPDB.csv")
df.set_index("protein_code", inplace=True)
df = df.dropna()
df = df.drop(df.columns[:1], axis=1)
"""threshold = 0.9
# Calculate the proportion of zeros in each column
zero_proportion = (df == 0).sum() / len(df)
# Drop columns where the proportion of zeros exceeds the threshold
columns_to_drop = zero_proportion[zero_proportion > threshold].index
df_filtered = df.drop(columns=columns_to_drop)"""
correlation = df.corr()
p_values = df.dropna().corr(method=lambda x,y: pearsonr(x,y)[1])
#correlation.to_excel("C:/Users/32496/OneDrive - KU Leuven/kuleuven(1)/IBP/correlation.xlsx", index=True)
significant_correlations = correlation[p_values <0.05]
#print(significant_correlations)
sns.heatmap(correlation, vmin=-1, vmax=1, cmap="rocket_r", cbar=False, xticklabels=False, yticklabels=False)
plt.savefig('heatmap.svg', format='svg', transparent=True)
plt.show()

