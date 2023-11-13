import pandas as pd
import re
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import ConvexHull

# Perform surface reconstruction and interpolation as needed


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
point_cloud=df[["x", "y", "z"]].to_numpy()

#make mesh for covering surface
tri = Delaunay(point_cloud)
boundary_points = point_cloud[tri.convex_hull]

x_data = boundary_points[:,0]
y_data = boundary_points[:,1]
z_data = boundary_points[:,2]

# Combine the x, y, and z data into a 3D point cloud
point_cloud = np.vstack((x_data, y_data, z_data)).T


# Visualize the reconstructed mesh (optional)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for simplex in tri.simplices:
    vertices = point_cloud[simplex]
    vertices = np.vstack((vertices, vertices[0]))  # Close the triangle
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 'b-')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()