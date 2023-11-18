import pandas as pd
import re
import numpy as np
from scipy.spatial import ConvexHull
from scipy.special import sph_harm
from math import factorial

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

x_data = df["x"].to_numpy()
y_data = df["y"].to_numpy()
z_data = df["z"].to_numpy()
point_cloud = df[["x", "y", "z"]].to_numpy()

#make mesh for covering surface
hull = ConvexHull(point_cloud)
boundary_points = point_cloud[hull.vertices]

def cartesian_to_spherical(cartesian_points):
    x, y, z = cartesian_points[:, 0], cartesian_points[:, 1], cartesian_points[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi

def compute_3d_zernike_descriptor(cartesian_boundary_points, max_degree):
    r, theta, phi = cartesian_to_spherical(cartesian_boundary_points)
    descriptors = []

    for degree in range(max_degree + 1):
        for order in range(-degree, degree + 1):
            descriptor = 0.0

            for i in range(len(cartesian_boundary_points)):
                radial_term = np.sqrt((2 * degree + 1) / (4 * np.pi)) * sph_harm(order, degree, theta[i], phi[i])
                normalization_term = np.sqrt(factorial(degree - abs(order)) / factorial(degree + abs(order)))

                descriptor += normalization_term * radial_term

            descriptor *= np.sqrt((2 * degree + 1) / (4 * np.pi))

            descriptors.append(descriptor)

    return descriptors

# Example usage
# Assuming 'boundary_points' is a NumPy array of boundary points in Cartesian coordinates
max_degree = 3
zernike_descriptors = compute_3d_zernike_descriptor(boundary_points, max_degree)
print(zernike_descriptors)