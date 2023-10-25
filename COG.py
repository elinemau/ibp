import pandas as pd
import re

#function to remove empty strings from list
def remove_emptys(string):
    return string !=""

#making dataframe to make x y z coordinates accessible
df = pd.DataFrame(columns=["number", "atom", "x", "y", "z"])
with open("1a28\\volsite\\ligand.mol2", "r") as file:
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

#calculate the center of gravity (COG)
#for ligand (22.844, 10.457, 60.069)
#for cavity1 (23.0199, 10.064, 59.941)  distance from prot mean = 10.45
#for cavity2 (21.763, 8.521, 70.01) distance from prot mean = 8.24

COG = df[["x","y","z"]].mean()
print(COG)
