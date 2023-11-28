import pandas as pd
import numpy as np
from itertools import combinations, combinations_with_replacement, product
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import os
from scipy.special import sph_harm
from math import factorial
from pymol import cmd


def load_mol_file(filename):
    """
    Loads the descriptors for a given cavity returned by Volsite into a pandas DataFrame.

    :param filename: str
        File name with Volsite descriptors (.txt).

    :param cavity_index: int
        Index of the selected cavity (the one closer to the ligand).

    :return: pandas.DataFrame or None
        DataFrame with Volsite descriptors for a given cavity, returns None if unsuccessful.
        """
    # check if the file is not (almost) empty
    f = open(filename, 'r')
    data = f.read().strip()
    if len(data) < 27:
        return None

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


def calculate_nearest_point(df_points, reference_point):
    """
    Calculates the index of the column in 'df_points' that is closest to the 'reference_point'.

    :param df: columns are point, row 1=x, row 2=y, row3=z
    :param reference: the reference point with same structure as df
    :return: the index of the column in df that is closest to reference
    """
    distance = []
    for column in df_points:
        d = math.sqrt((reference_point.iloc[0] - df_points[column].iloc[0]) ** 2 + (
                reference_point.iloc[1] - df_points[column].iloc[1]) ** 2 + (
                              reference_point.iloc[2] - df_points[column].iloc[2]) ** 2)
        distance.append(d)
    return distance.index(min(distance))


def select_cavity(folder, ligand_file_path):
    """
    Investigates all cavities found with Volsite (with ligand restriction) and selects the one closest
    to the center of gravity of the ligand.

    :return: df of the cavity closest to the ligand
    """

    cog = pd.DataFrame()
    cavities = []
    files = []
    # select cavity files from the folder and put them in a list
    for file in os.listdir(folder):
        # include ALL in name, because N2, N4, N6,... are duplicate files
        if "CAVITY" and "ALL" in file:
            f = os.path.join(folder, file)
            # get df from file
            df = load_mol_file(f)
            df_points = get_points(df)
            # calculate center of gravity
            center = center_of_gravity(df_points)
            cog[file] = center
            cavities.append(df)
            files.append(file)
        else:
            continue
    # get df from file
    df_ligand = load_mol_file(ligand_file_path)
    # calculate center of gravity
    protein_center = center_of_gravity(get_points(df_ligand))

    # check if there are any cavities found by Volsite
    if len(cavities) > 0:
        # compare the distance from the cavities to the ligand and return the closest cavity
        index = calculate_nearest_point(cog, protein_center)
        return files[index], index, cavities[index]
    else:
        return None, None, None


def get_volsite_descriptors(volsite_folder, cavity_index):
    """
    Loads the descriptors for a given cavity returned by Volsite into a pandas dataframe.

    :param filename: file name with volsite descriptors (.txt)
    :param cavity_index: index of the selected cavity (the one closer to the ligand)
    :return: dataframe with volsite descriptors for a given cavity
    """
    # get the volsite descriptor file
    volsite_files = os.listdir(volsite_folder)
    filename = ""
    for file in volsite_files:
        if file.endswith('descriptor.txt'):
            filename = file
            break
    # add a failsafe for the incorrect cavity_num later!!!
    points = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
    column_names = ['volume'] + points
    for point in points:
        column_names += [f'{point}_below_40', f'{point}_between_40_50', f'{point}_between_50_60',
                         f'{point}_between_60_70',
                         f'{point}_between_70_80', f'{point}_between_80_90', f'{point}_between_90_100',
                         f'{point}_between_100_110', f'{point}_between_110_120', f'{point}_120']
    column_names += ['name']
    df = pd.read_csv(f'{volsite_folder}/{filename}', sep=" ", index_col=False, header=None, names=column_names)
    descriptors = df.loc[cavity_index, df.columns != 'name']
    return descriptors


def max_dist_cavity_points(cavity):
    """
    Calculates the maximum distance between every combination of two cavity points ('CZ', 'CA', 'O', 'OD1', 'OG', 'N',
    'NZ', 'DU') and returns the values as a pandas dataframe.

    :param cavity: pandas dataframe representing a cavity (contains information on atoms / cavity points)
    :return: pandas dataframe with maximum distances between pairs of cavity points
    """
    grouped_cavity = cavity.groupby('atom_name')

    point_types = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
    point_types.sort()

    max_distances = pd.DataFrame(0.0, columns=list(combinations_with_replacement(point_types, 2)), index=[0])

    for (atom_type1, group1), (atom_type2, group2) in combinations_with_replacement(grouped_cavity, 2):
        pair = (atom_type1, atom_type2)

        # Extract coordinates directly from the DataFrame
        coords1 = group1[['x', 'y', 'z']].values
        coords2 = group2[['x', 'y', 'z']].values

        # Calculate pairwise distances without using np.newaxis
        dist_matrix = np.linalg.norm(coords1[:, None, :] - coords2, axis=-1)

        max_distance = dist_matrix.max()

        # Update max_distances directly
        max_distances.at[0, pair] = max(max_distances.at[0, pair], max_distance)

    return max_distances


def distance(point1, point2):
    """
    Calculates the distance between two points in 3D space.

    :param point1: first point in 3D space
    :param point2: second point in 3D space
    :return: distance between point1 and point2
    """
    return math.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))


def is_valid_triangle(side1, side2, side3):
    """
    Checks if the given sides can form a valid triangle.

    :param side1: first side of the triangle
    :param side2: second side of the triangle
    :param side3: third side of the triangle
    :return: true if given sides can form a valid triangle, false otherwise
    """
    return (side1 + side2 > side3) and (side2 + side3 > side1) and (side3 + side1 > side2)


def calculate_triangle_area(coord1, coord2, coord3):
    """
    Calculates the area of a triangle using Heron's formula.

    :param coord1: coordinates of the first point in 3D space
    :param coord2: coordinates of the second point in 3D space
    :param coord3: coordinates of the third point in 3D space
    :return: the area of the triangle formed by the three given points or 0 if the points do not form a valid triangle
    """
    # Calculate all sides of the triangle
    side1 = distance(coord1, coord2)
    side2 = distance(coord2, coord3)
    side3 = distance(coord3, coord1)

    if is_valid_triangle(side1, side2, side3):
        # Semiperimeter
        s = (side1 + side2 + side3) / 2

        # Heron's formula
        area = math.sqrt(s * (s - side1) * (s - side2) * (s - side3))
        return area
    else:
        return 0


def max_triplet_area(cavity):
    """
    Calculates the maximum area of a triangle formed by every combination of three cavity points ('CZ', 'CA', 'O',
    'OD1', 'OG', 'N', 'NZ', 'DU') and returns the values as a pandas dataframe.

    :param cavity: pandas dataframe representing a cavity (contains information on atoms / cavity points)
    :return: pandas dataframe with maximum area of a triangle formed by triplets of cavity points
    """
    grouped_cavity = cavity.groupby('atom_name')
    point_types = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
    point_types.sort()

    # Note: function combinations instead of combinations_with_replacement is used to save computational time
    max_areas = pd.DataFrame(0.0, columns=list(combinations(point_types, 3)), index=[0])

    for triplet_combination in combinations(grouped_cavity, 3):
        triplet = tuple(sorted([atom_type for atom_type, _ in triplet_combination]))

        # Extract coordinates from the DataFrame
        all_coords = [group[['x', 'y', 'z']].values for _, group in triplet_combination]
        # Create triplets of coordinates
        coord_triplets = product(*all_coords)

        for triangle in coord_triplets:
            # Update max_areas directly
            max_areas.at[0, triplet] = max(max_areas.at[0, triplet], calculate_triangle_area(*triangle))

    return max_areas


def pc_retrieval(df):
    """
    Computes the first and second principal components from a pandas DataFrame.

    :param df: pandas dataframe output from load_mol_file
    :return: first and second principal component
    """
    point_cloud = df[["x", "y", "z"]].to_numpy()
    pca = PCA(n_components=3)
    pca.fit(point_cloud)
    return pca.components_


def convexhull(cavity_points):
    """
    Computes the convex hull of a set of points representing a cavity.

    :param cavity: cavity.mol2 file obtained with the function get_points()
    :return: convex hull of cavity
    """
    # make mesh for covering surface
    return ConvexHull(cavity_points)


def plot_cavity(cavity_points, hull, save_path):
    """
    Plot the 3D cavity and save it to a file. Doesn't return anything.

    :param cavity_points: pandas.DataFrame obtained with the function get_points()
    :param hull: Convex hull retreived from convexHull def
    :param save_path: file path to save the generated plot.

    """
    plt.switch_backend('agg')  # Use the Agg backend

    # Select the boundary points using hull.vertices
    boundary_points = cavity_points_df.iloc[hull.vertices].to_numpy()
    # Create a figure and a subplot for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot the boundary points in red
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
               c='r', marker='o', label='Boundary Points')
    # Create a Poly3DCollection for the mesh using hull.simplices
    mesh = Poly3DCollection([cavity_points[s] for s in hull.simplices], alpha=0.25, edgecolor='k')
    # Add the mesh to the plot
    ax.add_collection3d(mesh)
    # Return the figure
    plt.savefig(save_path)  # Save the plot to a file
    plt.close(fig)  # Close the figure to release memory

def area(hull):
    """
    Calculates the surface area of a cavity represented by a convex hull.

    :param hull: convex hull retrieved from convexhull def from cavity
    :return: area of surface of cavity
    """
    surface_area = hull.area
    return surface_area


def distances_angles_shell_center(cavity_points, hull):
    """
    Computes the longest and shortest distance from the center to the surface of a cavity, along with the angle between them.

    :param cavity_points: pandas.DataFrame containing points representing the cavity, obtained from a cavity.mol2 file and def get_points()
    :param hull: scipy.spatial.ConvexHull: Convex hull object representing the surface of the cavity.
    :return: tuple: containing the shortest distance, longest distance, and angle between center and surface.
    """

    # compute euclidean distances from center to all points
    cavity_np = cavity_points.to_numpy()
    # find center
    center = center_of_gravity(cavity_points).to_numpy()
    boundary_points = cavity_np[hull.vertices]

    # compute distences between center and boundary points
    distances = np.linalg.norm(boundary_points - center, axis=1)

    # find the indices of the furthest and closest point
    furthest_point_index = np.argmax(distances)
    closest_point_index = np.argmin(distances)

    # get coordinates of the furthest and closest point
    furthest_point = boundary_points[furthest_point_index]
    closest_point = boundary_points[closest_point_index]

    # calculate distance to the furthest and closest points
    distance_to_furthest_point = distances[furthest_point_index]
    distance_to_closest_point = distances[closest_point_index]

    # calculate angle between the two
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

    return distance_to_closest_point, distance_to_furthest_point, angle_degrees


def cartesian_to_spherical(cartesian_boundary_points):
    """
    Converts Cartesian coordinates to spherical coordinates for a set of boundary points.

    :param cartesian_boundary_points: numpy.ndarray, Array containing Cartesian coordinates of boundary points (x, y, z).
    :return: tuple containing the converted spherical coordinates (r, theta, phi).
        """
    x, y, z = cartesian_boundary_points[:, 0], cartesian_boundary_points[:, 1], cartesian_boundary_points[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / r)
    return r, theta, phi


def compute_3d_descriptor(boundary_points, max_degree):
    """
    Computes 3D spherical harmonic descriptors for a set of boundary points.

    :param boundary_points: numpy.ndarray: Array containing the boundary points in Cartesian coordinates (x, y, z).
    :param max_degree: int: Maximum degree for the spherical harmonic descriptor computation.
    :return: List of computed spherical harmonic descriptors."""

    r, theta, phi = cartesian_to_spherical(boundary_points)
    descriptors = []
    for degree in range(max_degree + 1):
        for order in range(-degree, degree + 1):
            descriptor = 0.0
            for i in range(len(boundary_points)):
                radial_term = np.sqrt((2 * degree + 1) / (4 * np.pi)) * sph_harm(order, degree, theta[i], phi[i])
                normalization_term = np.sqrt(factorial(degree - abs(order)) / factorial(degree + abs(order)))
                descriptor += normalization_term * radial_term
            descriptor *= np.sqrt((2 * degree + 1) / (4 * np.pi))
            descriptors.append(descriptor)
    return descriptors


def find_neighboring_residues(protein_file, cavity_file, distance_threshold=4.0):
    """
    Identify and retrieve the residue indices of atoms within a specified distance threshold from a cavity within a
    protein structure.

    :param protein_file: str, The file path to the protein structure in a format compatible with PyMOL.
    :param cavity_file: str, The file path to the cavity structure in a format compatible with PyMOL.
    :param distance_threshold: float, optional, The distance threshold (in angstroms) used to filter atoms within the
        cavity. Defaults to 4.0 angstroms.

    :return: set, A set containing the residue indices of atoms within the specified distance threshold from the cavity.
    """
    # Load ligand and protein in PyMOL
    cmd.load(protein_file)
    cmd.load(cavity_file)

    cavity_obj = cavity_file.split('/')[-1].split('.')[0]

    # Select the object by name
    selection_name = 'cavity_atoms'
    cmd.select(selection_name, cavity_obj)

    # Modify the selection to include residues within the distance threshold
    cmd.select(selection_name, f'{selection_name} around {distance_threshold}')

    # Print the residue numbers in the modified selection
    model = cmd.get_model(selection_name)
    res_lim = model.get_residues()

    atom_list = model.atom
    resid_indices = set()

    for start, end in res_lim:  # extract the data we are interested in
        for atom in atom_list[start:end]:
            resid_indices.add(atom.resi)

    return resid_indices


def is_residue_exposed_to_cavity(protein, cavity, residue_id, dot_product_threshold=0.0):
    """
    Determine whether a residue is exposed to a cavity in a protein structure based on the cosine of angles.

    :param protein: DataFrame, The protein structure data containing information about atoms.
    :param cavity: DataFrame, The cavity structure data containing information about atoms.
    :param residue_id: int, The identifier of the residue to be checked for exposure.
    :param dot_product_threshold: float, optional, The threshold for the cosine of angles to consider a residue exposed.
                                Defaults to 0.0.

    :return: tuple (bool, str or None), A tuple indicating whether the residue is exposed and, if so, whether it is the
        'side_chain' or 'backbone'.
    """
    cavity_center = center_of_gravity(get_points(cavity))

    residue = protein[protein['subst_name'].str.endswith(str(residue_id))]

    # Calculate the vector between the residue's backbone (N, CA, C) and the cavity's center of gravity
    backbone_atoms = ['N', 'CA', 'C', 'O']
    backbone = pd.concat([residue[residue['atom_name'] == atom] for atom in backbone_atoms], ignore_index=True)
    backbone_center = center_of_gravity(get_points(backbone))
    backbone_direction_vector = np.array(cavity_center - backbone_center)

    # Calculate the vector between the residue's side chain and the cavity's center of gravity
    side_chain_atoms = np.setdiff1d(np.unique(protein[['atom_name']].values), backbone_atoms)
    side_chain = pd.concat([residue[residue['atom_name'] == atom] for atom in side_chain_atoms], ignore_index=True)
    side_chain_center = center_of_gravity(get_points(side_chain))
    side_chain_direction_vector = np.array(cavity_center - side_chain_center)

    # Calculate the cosine of the angle between vectors
    backbone_cosine_angle = np.dot(backbone_direction_vector, cavity_center) / (
            np.linalg.norm(backbone_direction_vector) * np.linalg.norm(cavity_center))
    side_chain_cosine_angle = np.dot(side_chain_direction_vector, cavity_center) / (
            np.linalg.norm(side_chain_direction_vector) * np.linalg.norm(cavity_center))

    # Check if the cosine of the angles are greater than the threshold to determine exposure
    if side_chain_cosine_angle > dot_product_threshold:
        return True, 'side_chain'
    elif backbone_cosine_angle > dot_product_threshold:
        return True, 'backbone'
    else:
        return False, None


def get_exposed_residues(protein_file, protein, cavity_file, cavity, distance_threshold=4.0,
                         dot_product_threshold=0.0):
    """
    Analyze and classify exposed residues surrounding a cavity in a protein structure.

    :param protein_file: str, The file path to the protein structure in a format compatible with PyMOL.
    :param protein: DataFrame, The protein structure data containing information about atoms.
    :param cavity_file: str, The file path to the cavity structure in a format compatible with PyMOL.
    :param cavity: DataFrame, The cavity structure data containing information about atoms.
    :param distance_threshold: float, optional, The distance threshold (in angstroms) for identifying neighboring residues.
                              Defaults to 4.0 angstroms.
    :param dot_product_threshold: float, optional, The threshold for the cosine of angles to consider a residue exposed.
                                Defaults to 0.0.

    :return: DataFrame, A DataFrame containing information about exposed residues.
    """
    # Get the residues that surround the cavity
    neighboring_residues = find_neighboring_residues(protein_file, cavity_file)

    exposed_backbone = 0
    exposed_side_chain = 0
    polar_side_chain = 0
    aromatic_side_chain = 0
    pos_side_chain = 0
    neg_side_chain = 0
    hydrophobic_side_chain = 0

    # Check each residue
    for residue in neighboring_residues:
        result, exposed_part = is_residue_exposed_to_cavity(protein, cavity, residue)
        if exposed_part == 'backbone':
            exposed_backbone += 1
        elif exposed_part == 'side_chain':
            exposed_side_chain += 1
            res = protein['subst_name'][protein['subst_name'].str.endswith(str(residue))].values[0]
            resn = res[:3]
            # Check if the side chain is polar
            if resn in ['SER', 'THR', 'CYS', 'PRO', 'ASN', 'GLN']:
                polar_side_chain += 1
            # Check if the side chain is aromatic
            if resn in ['PHE', 'TRP', 'TYR']:
                aromatic_side_chain += 1
            # Check if the side chain is positive
            if resn in ['LYS', 'ARG', 'HIS']:
                pos_side_chain += 1
            # Check if the side chain is negative
            if resn in ['ASP', 'GLU']:
                neg_side_chain += 1
            # Check if the side chain is hydrophobic
            if resn in ['GLY', 'PRO', 'PHE', 'ALA', 'ILE', 'LEU', 'VAL']:
                hydrophobic_side_chain += 1

    all_exposed = exposed_backbone + exposed_side_chain

    exposed_residues = {'exposed_residues': all_exposed,
                        'exposed_backbone_abs': exposed_backbone,
                        'exposed_backbone_ratio_all': float(exposed_backbone / all_exposed) if all_exposed > 0 else 0.0,
                        'exposed_side_chain_abs': exposed_side_chain,
                        'exposed_side_chain_ratio_all': float(exposed_side_chain /
                                                              all_exposed) if all_exposed > 0 else 0.0,
                        'exposed_polar_side_abs': polar_side_chain,
                        'exposed_polar_side_ratio': float(polar_side_chain /
                                                          exposed_side_chain) if exposed_side_chain > 0 else 0.0,
                        'exposed_aromatic_side_abs': aromatic_side_chain,
                        'exposed_aromatic_side_ratio': float(aromatic_side_chain /
                                                             exposed_side_chain) if exposed_side_chain > 0 else 0.0,
                        'exposed_pos_side_abs': pos_side_chain,
                        'exposed_pos_side_ratio': float(pos_side_chain /
                                                        exposed_side_chain) if exposed_side_chain > 0 else 0.0,
                        'exposed_neg_side_abs': neg_side_chain,
                        'exposed_neg_side_ratio': float(neg_side_chain /
                                                        exposed_side_chain) if exposed_side_chain > 0 else 0.0,
                        'exposed_hydrophobic_side_abs': hydrophobic_side_chain,
                        'exposed_hydrophobic_side_ratio': float(hydrophobic_side_chain /
                                                                exposed_side_chain) if exposed_side_chain > 0 else 0.0
                        }

    df_exposed = pd.DataFrame(exposed_residues, index=[0])
    return df_exposed


def get_directory_input():
    """
    Requests user input for a directory path until a valid directory path is provided.

    :return: str: Valid directory path obtained from user input.
    :return:
    """
    while True:
        directory_path = input("Enter a directory path: ")
        # Check if the provided path is a directory
        if os.path.isdir(directory_path):
            return directory_path
        else:
            print("Invalid directory. Please enter a valid directory path.")


def list_subdirectories(directory):
    """
    Retrieves a list of subdirectories within the specified directory.

    :param directory: Path to the directory.
    :return: list of subdirectory paths found within the specified directory.
    """
    # Get the list of files in the directory
    input_proteins_list = []
    subdirs = os.listdir(directory)
    # Iterate over each file
    for subd in subdirs:
        # Get the full path of the file
        subd_path = os.path.join(directory, subd)
        # Check if the path is a subdir
        if os.path.isdir(subd_path):
            input_proteins_list.append(subd_path)
    return input_proteins_list


if __name__ == '__main__':
    # algorith is run from the directory containing the 3 directories (01_removed_waters_pdb, 02_input_files_mol,
    # 03_volsite)
    input_proteins = list_subdirectories("03_volsite")
    all_descriptors = pd.DataFrame()

    # Create a new directory to store the figures
    figures_folder_name = '04_figures'
    # Check if the folder doesn't exist, then create it
    if not os.path.exists(figures_folder_name):
        os.makedirs(figures_folder_name)
        print(f"Folder '{figures_folder_name}' created successfully.")
    else:
        print(f"Folder '{figures_folder_name}' already exists.")

    tmp = 1

    for protein_volsite in input_proteins:
        print(tmp)
        protein_code = protein_volsite.split('\\')[-1]
        print(protein_code)

        protein_path = f'02_input_files_mol/{protein_code}.mol2'
        ligand_path = f'02_input_files_mol/{protein_code}_lig.mol2'
        # Select the cavity that covers the ligand
        cavity_file, cavity_index, cavity_df = select_cavity(protein_volsite, ligand_path)
        cavity_path = f'03_volsite/{protein_code}/{cavity_file}'

        if cavity_df is not None and load_mol_file(protein_path) is not None:
            print(cavity_df.shape[0])
            # Get the descriptors generated by Volsite
            volsite_descriptors = get_volsite_descriptors(protein_volsite, cavity_index)

            cavity_descriptors = pd.DataFrame(volsite_descriptors).transpose()
            cavity_descriptors = cavity_descriptors.set_axis([0], axis=0)
            cavity_descriptors.insert(0, "protein_code", protein_code)

            # Cavity X, Y, Z coordinates
            cavity_points_df = get_points(cavity_df)
            # Add cavity area to the df
            hull = convexhull(cavity_points_df)
            cavity_area = area(hull)
            cavity_descriptors = cavity_descriptors.assign(area=[cavity_area])

            # Add min & max distance to the center of the gravity & shell & the angle between those
            distance_to_closest_point, distance_to_furthest_point, angle_degrees = (
                distances_angles_shell_center(cavity_points_df, hull))
            cavity_descriptors = cavity_descriptors.assign(min_dist=[distance_to_closest_point])
            cavity_descriptors = cavity_descriptors.assign(max_dist=[distance_to_furthest_point])
            cavity_descriptors = cavity_descriptors.assign(angle=[angle_degrees])

            # Add max dist between two cavity points to the descriptors df
            max_dist_pairs = max_dist_cavity_points(cavity_df)
            cavity_descriptors = pd.concat([cavity_descriptors, max_dist_pairs], axis=1)

            # Add max area of a triangle formed by three cavity points
            max_area_triplets = max_triplet_area(cavity_df)
            cavity_descriptors = pd.concat([cavity_descriptors, max_area_triplets], axis=1)

            # Load the protein mol2 file
            protein_df = load_mol_file(protein_path)
            # Get residues exposed to the cavity
            exposed_aa = get_exposed_residues(protein_path, protein_df, cavity_path, cavity_df)
            cavity_descriptors = pd.concat([cavity_descriptors, exposed_aa], axis=1)

            # Make the plot and save it to a file in '04_figures' directory
            save_path = f'04_figures/{protein_code}_plot.png'
            plot_cavity(cavity_points_df.to_numpy(), hull, save_path)

            # Add the descriptors of the current cavity to the general dataframe
            if all_descriptors.empty:
                all_descriptors = cavity_descriptors
            else:
                all_descriptors = pd.concat([all_descriptors, cavity_descriptors.iloc[[-1]]], ignore_index=True)

        else:
            no_cavity = pd.DataFrame()
            no_cavity = no_cavity.set_axis([0], axis=0)
            no_cavity.insert(0, "protein_code", protein_code)
            print(no_cavity)
            all_descriptors = pd.concat([all_descriptors, no_cavity.iloc[[-1]]], ignore_index=True)
            print("all_descriptors \n", all_descriptors)
        tmp += 1
        all_descriptors.to_csv(f'all_descriptors.csv')

    all_descriptors.to_csv(f'all_descriptors.csv')
