import numpy as np
from scipy.spatial import ConvexHull
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
from matplotlib.colors import LightSource, Normalize


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


def center_of_gravity(points):
    """

    :param points: coordinates retreived from get_points
    :return: center of gravity for the given structure
    """
    # Calculate the center of gravity of the structure
    return points.mean()


def sphericity(points, elev_init=20, azim_init=30):
    points = points.to_numpy()
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
    # make fig to plot  mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the sphere
    phi, theta = np.mgrid[0.0:2.0 * np.pi:20j, 0.0:np.pi:10j]
    x = sphere_radius * np.sin(theta) * np.cos(phi)
    y = sphere_radius * np.sin(theta) * np.sin(phi)
    z = sphere_radius * np.cos(theta)
    # Calculate the convex hull for the sphere
    sphere_points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
    sphere_hull = ConvexHull(sphere_points)
    # Plot the convex hull mesh for the sphere
    sphere_mesh = Poly3DCollection([sphere_points[s] for s in sphere_hull.simplices], alpha=0.2, edgecolor='grey',
                                   facecolors='white')

    ax.add_collection3d(sphere_mesh)
    mesh = Poly3DCollection([points[s] for s in hull.simplices], alpha=0.4, edgecolor='lightgray',
                            facecolors="orange")
    ax.add_collection3d(mesh)
    ax.set_ylim(-6,6)
    ax.set_xlim(-6,6)
    ax.set_zlim(-6,6)
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('none')
    ax.view_init(elev=20, azim=30)

    def update_view(elev, azim):
        ax.view_init(elev=elev_init, azim=azim_init)
        plt.draw()

    # Create an interactive slider for elevation angle
    elev_slider = plt.Slider(ax=plt.axes([0.1, 0.01, 0.65, 0.03]), label='Elevation', valmin=0, valmax=90,
                             valinit=elev_init)
    elev_slider.on_changed(lambda elev: update_view(elev, azim_slider.val))

    # Create an interactive slider for azimuthal angle
    azim_slider = plt.Slider(ax=plt.axes([0.1, 0.06, 0.65, 0.03]), label='Azimuth', valmin=0, valmax=360,
                             valinit=azim_init)
    azim_slider.on_changed(lambda azim: update_view(elev_slider.val, azim))
    ax.view_init(elev=1, azim=42)
    plt.savefig('sphere_shape.svg', format='svg', transparent=True)
    plt.show()

    return sphericity


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


def plot_obb(points, principal_axes, centroid, elev_init=20, azim_init=30):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the original points
    if isinstance(points, np.ndarray):
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='gray', marker='o', label='Original Points')
    elif isinstance(points, pd.DataFrame):
        ax.scatter(points['x'], points['y'], points['z'], c='gray', marker='o', label='Original Points')

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

    ax.add_collection3d(Poly3DCollection(box, facecolors='orange', linewidths=1, edgecolors='gray', alpha=.25))
    ax.grid(False)
    ax.axis('off')
    ax.set_facecolor('none')
    ax.view_init(elev=35, azim=15)
    plt.savefig('smallest_box.svg', format='svg', transparent=True)
    plt.show()


if __name__ == '__main__':
    cavity_file = load_mol_file("1a28\\1a28\\CAVITY_N1_ALL.mol2")
    cavity_points = get_points(cavity_file)
    cavity_center = center_of_gravity(cavity_points)
    centered_points = center_points(cavity_points)
    #sphere = sphericity(centered_points)
    principal_axes, centroid = find_obb(centered_points)
    plot_obb(centered_points, principal_axes, centroid)
    #Calculate the extent along each principal axis
    #extent = np.max(centered_points.dot(principal_axes.T), axis=0) - np.min(centered_points.dot(principal_axes.T),axis=0)
    #extent vector contains length, width and hight of the smallest oriented bounding box
    #print(extent)
