
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA

#making dataframe to make x y z coordinates accessible
df = pd.DataFrame(columns=["number", "atom", "x", "y", "z"])
with open("1a28\\1a28\\CAVITY_N1_ALL.mol2", "r") as file:
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

x_data = df["x"].to_numpy()
y_data = df["y"].to_numpy()
z_data = df["z"].to_numpy()
point_cloud = df[["x", "y", "z"]].to_numpy()

#make mesh for covering surface
hull = ConvexHull(point_cloud)
boundary_points = point_cloud[hull.vertices]
print(boundary_points)

#make fig to plot  mesh
fig=plt.figure()
ax=fig.add_subplot(111, projection='3d')
"""ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2], c='r', marker='o', label='Boundary Points')
ax.scatter(point_cloud[:,0], point_cloud[:,1], point_cloud[:,2], c='b', marker='o', label='grid')"""
mesh = Poly3DCollection([point_cloud[s] for s in hull.simplices], alpha=0.25, facecolors='cornflowerblue', linewidths=1, edgecolors='lightsteelblue')
ax.add_collection3d(mesh)

#find furthest point in cloud from center
#first find center
center = np.mean(point_cloud, axis=0)
# compute euclidean distances from center to all points
distances = np.linalg.norm(boundary_points-center, axis=1)
furthest_point_index = np.argmax(distances)
furthest_point = boundary_points[furthest_point_index]
distance_to_furthest_point = distances[furthest_point_index]
closest_point_index = np.argmin(distances)
closest_point = boundary_points[closest_point_index]
distance_to_closest_point = distances[closest_point_index]
#calculate angle between the two
# Calculate the vectors from the center to the closest and furthest points
closest_point_vector = closest_point - center
furthest_point_vector = furthest_point - center
# Calculate the dot product between the two vectors
dot_product = np.dot(closest_point_vector, furthest_point_vector)
# Calculate the magnitudes (lengths) of the vectors
closest_point_magnitude = np.linalg.norm(closest_point_vector)
furthest_point_magnitude = np.linalg.norm(furthest_point_vector)
# Calculate the angle in radians using the dot product and magnitudes
angle_radians = np.arccos(dot_product / (closest_point_magnitude * furthest_point_magnitude))
# Convert the angle from radians to degrees
angle_degrees = np.degrees(angle_radians)

print(f"Angle between closest and furthest points: {angle_degrees:.2f} degrees")

#plot line to furthest point
ax.plot([center[0], furthest_point[0]], [center[1], furthest_point[1]], [center[2], furthest_point[2]], 'g-', label='distance line')
ax.plot([center[0], closest_point[0]], [center[1], closest_point[1]], [center[2], closest_point[2]], 'g-', label='distance line')

ax.set_xlim(30,40)
ax.set_ylim(30, 40)
ax.set_zlim(35, 50)
ax.grid(False)
ax.axis('off')
ax.set_facecolor('none')
plt.show()


