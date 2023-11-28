import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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


def rotating_calipers(points):
    hull = ConvexHull(points)
    convex_hull_points = points.values[hull.vertices]  # Extract points on the convex hull

    min_width = float('inf')
    min_width_points = None

    for i in range(len(convex_hull_points)):
        p1 = convex_hull_points[i]
        p2 = convex_hull_points[(i + 1) % len(convex_hull_points)]

        # Compute the direction vector of the current edge
        edge_vector = p2 - p1

        # Normalize the edge vector
        edge_vector /= np.linalg.norm(edge_vector)

        # Construct the 3D rotation matrix for the current edge
        rotation_matrix = np.array([[edge_vector[0], edge_vector[1], 0],
                                    [-edge_vector[1], edge_vector[0], 0],
                                    [0, 0, 1]])

        # Rotate the convex hull points to align with the current edge
        rotated_points = convex_hull_points.dot(rotation_matrix.T)

        # Compute the minimum and maximum x-coordinates of the rotated points
        min_x = np.min(rotated_points[:, 0])
        max_x = np.max(rotated_points[:, 0])

        # Compute the width of the bounding box
        width = max_x - min_x

        # Update the minimum width if necessary
        if width < min_width:
            min_width = width
            min_width_points = rotated_points

    return min_width, min_width_points

def plot_rotated_calipers_as_box(points, min_width_points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot original points
    ax.scatter(points['x'], points['y'], points['z'], c='blue', marker='o', label='Original Points')

    # Plot rotated bounding box
    edges = [
        [min_width_points[0], min_width_points[1], min_width_points[2], min_width_points[3], min_width_points[0]],
        [min_width_points[4], min_width_points[5], min_width_points[6], min_width_points[7], min_width_points[4]],
        [min_width_points[0], min_width_points[1], min_width_points[5], min_width_points[4], min_width_points[0]],
        [min_width_points[2], min_width_points[3], min_width_points[7], min_width_points[6], min_width_points[2]],
        [min_width_points[1], min_width_points[2], min_width_points[6], min_width_points[5], min_width_points[1]],
        [min_width_points[0], min_width_points[3], min_width_points[7], min_width_points[4], min_width_points[0]]
    ]

    poly3d = Poly3DCollection(edges, facecolors='red', linewidths=1, edgecolors='r', alpha=0.2)
    ax.add_collection3d(poly3d)

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
    print(cone, sphere, cube)
    centered_points = center_points(cavity_points)
    mid_width, min_width_points = rotating_calipers(centered_points)
    calplot = plot_rotated_calipers(centered_points, min_width_points)
    calplot.show()