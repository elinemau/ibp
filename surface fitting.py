#scikit-learn, pykrige

#pykrige
import numpy as np
import pandas as pd
import re
from scipy.spatial import ConvexHull
from pykrige.ok import OrdinaryKriging

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

#make mesh for covering surface
hull = ConvexHull(point_cloud)
boundary_points = point_cloud[hull.vertices]

x_data = boundary_points[:,0]
y_data = boundary_points[:,1]
z_data = boundary_points[:,2]

#define grid for interpolation
x_min, x_max = x_data.min(), x_data.max()
y_min, y_max = y_data.min(), y_data.max()
grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20))

#create ordinary kriging model
OK = OrdinaryKriging(
    x_data,
    y_data,
    z_data,
    variogram_model='linear',  # You can choose a different variogram model
    verbose=False,
    enable_plotting=False
)

#interpolate the surface
z_interpolated, _ = OK.execute('grid', grid_x, grid_y)

#visualize the interpolated surface
import matplotlib.pyplot as plt

plt.contourf(grid_x, grid_y, z_interpolated, levels=100, cmap='viridis')
plt.colorbar()
plt.scatter(x_data, y_data, c=z_data, cmap='viridis', edgecolors='k')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolated Surface')
plt.show()
