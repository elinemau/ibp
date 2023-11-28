import csv
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_excel("C:/Users/32496/OneDrive - KU Leuven/kuleuven(1)/IBP/dataframe.xlsx")
df.set_index("protein_code", inplace=True)
correlation = df.dropna().corr()
p_values = df.dropna().corr(method=lambda x,y: pearsonr(x,y)[1])
correlation.to_excel("C:/Users/32496/OneDrive - KU Leuven/kuleuven(1)/IBP/correlation.xlsx", index=True)
p_values.to_excel("C:/Users/32496/OneDrive - KU Leuven/kuleuven(1)/IBP/p_values.xlsx")
print(p_values)
sns.heatmap(correlation, vmin=-1, vmax=1, cmap="rocket_r")
plt.show()

