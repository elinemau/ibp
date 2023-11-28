import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA


def load_mol_file(filename):
    df = pd.DataFrame(columns=['atom_id', 'atom_name', 'x', 'y', 'z', 'atom_type', 'subst_id', 'subst_name', 'charge'])
    with open(filename, "r") as file:
        line = file.readline()
        while not line.startswith("@<TRIPOS>ATOM"):
            line = file.readline()
        line = file.readline()
        while not line.startswith("@<TRIPOS>BOND"):
            data = line.strip().split()
            # Convert 'x', 'y', and 'z' columns to float
            data[2:5] = map(float, data[2:5])
            df.loc[len(df)] = data
            line = file.readline()
    file.close()
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    df['charge'] = df['charge'].astype(float)
    return df

def get_points(df):
    """

    :param df: mol2 file processed by load_mol2_file
    :return: coordinates
    """
    return df[["x", "y", "z"]]


def sphericity(points):
    # Calculate the convex hull
    hull = ConvexHull(points)

    # Calculate the volume of the convex hull
    hull_volume = hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = hull.area

    # Calculate the radius of a sphere with the same volume
    sphere_radius = ((3 * hull_volume) / (4 * np.pi))**(1/3)

    # Calculate the surface area of a sphere with the same volume
    sphere_surface_area = 4 * np.pi * (sphere_radius**2)

    # Calculate the sphericity
    sphericity = hull_surface_area / sphere_surface_area

    return sphericity

def cubic_sphericity(points):
    # Calculate the convex hull
    hull = ConvexHull(points)

    # Calculate the volume of the convex hull
    hull_volume = hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = hull.area

    # Calculate the side length of a cube with the same volume
    cube_side_length = (hull_volume)**(1/3)

    # Calculate the surface area of a cube with the same volume
    cube_surface_area = 6 * (cube_side_length**2)

    # Calculate the cubic sphericity
    cubic_sphericity = hull_surface_area / cube_surface_area

    return cubic_sphericity


def cone_sphericity(points):
    # Calculate the convex hull
    hull = ConvexHull(points)

    # Calculate the volume of the convex hull
    hull_volume = hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = hull.area

    # Calculate the radius and height of a cone with the same volume
    cone_radius = np.sqrt(hull_surface_area / (np.pi * hull_volume))
    cone_height = hull_volume / (np.pi * cone_radius**2)

    # Calculate the surface area of a cone with the same volume
    cone_surface_area = np.pi * cone_radius * (cone_radius + np.sqrt(cone_radius**2 + cone_height**2))

    # Calculate the cone sphericity
    cone_sphericity = hull_surface_area / cone_surface_area

    return cone_sphericity


def center_points(points):
    means = points.mean()
    return points-means

def find_obb(points):
    # Apply PCA to find principal components and directions
    pca = PCA(n_components=3)
    pca.fit(points)

    # The components_ attribute contains the principal axes
    principal_axes = pca.components_

    # The mean_ attribute contains the centroid of the points
    centroid = pca.mean_

    return principal_axes, centroid

def plot_obb(points, principal_axes, centroid):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original points
    if isinstance(points, np.ndarray):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', marker='o', label='Original Points')
    elif isinstance(points, pd.DataFrame):
        ax.scatter(points['x'], points['y'], points['z'], c='blue', marker='o', label='Original Points')

    # Plot the principal axes
    for axis in principal_axes:
        axis_line = np.vstack([centroid, centroid + axis])
        ax.plot3D(axis_line[:, 0], axis_line[:, 1], axis_line[:, 2], c='red', linewidth=2)

    # Calculate the extent along each principal axis
    extent = np.max(points.dot(principal_axes.T), axis=0) - np.min(points.dot(principal_axes.T), axis=0)

    # Create a Poly3DCollection representing the bounding box
    box_vertices = np.array([
        centroid - 0.5 * principal_axes[0] * extent[0] - 0.5 * principal_axes[1] * extent[1] - 0.5 * principal_axes[2] * extent[2],
        centroid + 0.5 * principal_axes[0] * extent[0] - 0.5 * principal_axes[1] * extent[1] - 0.5 * principal_axes[2] * extent[2],
        centroid + 0.5 * principal_axes[0] * extent[0] + 0.5 * principal_axes[1] * extent[1] - 0.5 * principal_axes[2] * extent[2],
        centroid - 0.5 * principal_axes[0] * extent[0] + 0.5 * principal_axes[1] * extent[1] - 0.5 * principal_axes[2] * extent[2],
        centroid - 0.5 * principal_axes[0] * extent[0] - 0.5 * principal_axes[1] * extent[1] + 0.5 * principal_axes[2] * extent[2],
        centroid + 0.5 * principal_axes[0] * extent[0] - 0.5 * principal_axes[1] * extent[1] + 0.5 * principal_axes[2] * extent[2],
        centroid + 0.5 * principal_axes[0] * extent[0] + 0.5 * principal_axes[1] * extent[1] + 0.5 * principal_axes[2] * extent[2],
        centroid - 0.5 * principal_axes[0] * extent[0] + 0.5 * principal_axes[1] * extent[1] + 0.5 * principal_axes[2] * extent[2]
    ])

    box = [[box_vertices[i] for i in [0, 1, 2, 3]],
           [box_vertices[i] for i in [4, 5, 6, 7]],
           [box_vertices[i] for i in [0, 3, 7, 4]],
           [box_vertices[i] for i in [1, 2, 6, 5]],
           [box_vertices[i] for i in [0, 1, 5, 4]],
           [box_vertices[i] for i in [2, 3, 7, 6]]]

    ax.add_collection3d(Poly3DCollection(box, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    cavity_file = load_mol_file("1a28\\volsite\\CAVITY_N1_ALL.mol2")
    cavity_points = get_points(cavity_file)
    cone = cone_sphericity(cavity_points)
    sphere = sphericity(cavity_points)
    cube = cubic_sphericity(cavity_points)
    centered_points = center_points(cavity_points)
    principal_axes, centroid = find_obb(centered_points)
    plot_obb(centered_points, principal_axes, centroid)
    # Calculate the extent along each principal axis
    extent = np.max(centered_points.dot(principal_axes.T), axis=0) - np.min(centered_points.dot(principal_axes.T),axis=0)
    #extent vector contains length, width and hight of the smallest oriented bounding box

