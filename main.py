import pandas as pd
import numpy as np
from itertools import combinations_with_replacement, product
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import os
from pymol import cmd
from sys import argv


def load_mol_file(filename):
    """
    Loads the descriptors for a given cavity returned by Volsite into a pandas DataFrame.

    :param filename: file name with Volsite descriptors (.txt).
    :return: pandas.DataFrame or None: DataFrame with Volsite descriptors for a given cavity, returns None if
        unsuccessful.
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

    :param points: coordinates retrieved from get_points
    :return: center of gravity for the given structure
    """
    # Calculate the center of gravity of the structure
    return points.mean()


def calculate_nearest_point(df_points, reference_point):
    """
    Calculates the index of the column in 'df_points' that is closest to the 'reference_point'.

    :param df_points: columns are point, row 1=x, row 2=y, row3=z
    :param reference_point: the reference point with same structure as df
    :return: the index of the column in df that is closest to reference
    """
    dist = []
    for column in df_points:
        d = math.sqrt((reference_point.iloc[0] - df_points[column].iloc[0]) ** 2 + (
                reference_point.iloc[1] - df_points[column].iloc[1]) ** 2 + (
                              reference_point.iloc[2] - df_points[column].iloc[2]) ** 2)
        dist.append(d)
    return dist.index(min(dist))


def select_cavity(folder, ligand_file_path):
    """
    Investigates all cavities found with Volsite (with ligand restriction) and selects the one closest
    to the center of gravity of the ligand.

    :param folder: folder with Volsite output
    :param ligand_file_path: path to the ligand file
    :return: dataframe of the cavity closest to the ligand
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


def get_volsite_descriptors(volsite_folder, cavity_ind):
    """
    Loads the descriptors for a given cavity returned by Volsite into a pandas dataframe.

    :param volsite_folder: folder with volsite output
    :param cavity_ind: index of the selected cavity (the one closer to the ligand)
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
                         f'{point}_between_60_70', f'{point}_between_70_80', f'{point}_between_80_90',
                         f'{point}_between_90_100', f'{point}_between_100_110', f'{point}_between_110_120',
                         f'{point}_120']
    column_names += ['name']
    df = pd.read_csv(f'{volsite_folder}/{filename}', sep=" ", index_col=False, header=None, names=column_names)
    descriptors = df.loc[cavity_ind, df.columns != 'name']
    return descriptors


def max_dist_cavity_points(pharmacophore):
    """
    Calculates the maximum distance between every combination of two pharmacophore points ('CZ', 'CA', 'O', 'OD1', 'OG',
    'N', 'NZ', 'DU') and returns the values as a pandas dataframe.

    :param pharmacophore: pandas dataframe representing a cavity (contains information on atoms / cavity points)
    :return: pandas dataframe with maximum distances between pairs of cavity points
    """
    grouped_pharmacophore = pharmacophore.groupby('atom_name')

    point_types = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
    point_types.sort()

    max_distances = pd.DataFrame(0.0, columns=list(combinations_with_replacement(point_types, 2)), index=[0])

    for (atom_type1, group1), (atom_type2, group2) in combinations_with_replacement(grouped_pharmacophore, 2):
        if atom_type1 not in point_types or atom_type2 not in point_types:
            continue
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
        triangle_area = math.sqrt(s * (s - side1) * (s - side2) * (s - side3))
        return triangle_area
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
    max_areas = pd.DataFrame(0.0, columns=list(combinations_with_replacement(point_types, 3)), index=[0])

    for triplet_combination in combinations_with_replacement(grouped_cavity, 3):
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

    :param cavity_points: cavity.mol2 file obtained with the function get_points()
    :return: convex hull of cavity
    """
    # make mesh for covering surface
    return ConvexHull(cavity_points)


def plot_cavity(cavity_points, convex_hull, path):
    """
    Plot the 3D cavity and save it to a file. Doesn't return anything.

    :param cavity_points: pandas.DataFrame obtained with the function get_points()
    :param convex_hull: Convex hull retrieved from convexHull def
    :param path: file path to save the generated plot.

    """
    # Select the boundary points using convex_hull.vertices
    boundary_points = cavity_points[convex_hull.vertices]
    # Turn of the interactive plotting, so the window with the plot doesn't pop up
    plt.switch_backend('agg')
    # Create a figure and a subplot for 3D plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot the boundary points in red
    ax.scatter(boundary_points[:, 0], boundary_points[:, 1], boundary_points[:, 2],
               c='r', marker='o', label='Boundary Points')
    # Create a Poly3DCollection for the mesh using convex_hull.simplices
    mesh = Poly3DCollection([cavity_points[s] for s in convex_hull.simplices], alpha=0.25, edgecolor='k')
    # Add the mesh to the plot
    ax.add_collection3d(mesh)
    # Return the figure
    plt.savefig(path)  # Save the plot to a file
    plt.close(fig)  # Close the figure to release memory


def area(convex_hull):
    """
    Calculates the surface area of a cavity represented by a convex hull.

    :param convex_hull: convex hull retrieved from convexhull def from cavity
    :return: area of surface of cavity
    """
    surface_area = convex_hull.area
    return surface_area


def distances_angles_shell_center(cavity_points, convex_hull):
    """
    Computes the longest and shortest distance from the center to the surface of a cavity, along with the angle between
    them.

    :param cavity_points: pandas.DataFrame containing points representing the cavity, obtained from a cavity.mol2 file
        and def get_points().
    :param convex_hull: scipy.spatial.ConvexHull: Convex hull object representing the surface of the cavity.
    :return: tuple: containing the shortest distance, the longest distance, and angle between center and surface.
    """

    # compute euclidean distances from center to all points
    cavity_np = cavity_points.to_numpy()
    # find center
    center = center_of_gravity(cavity_points).to_numpy()
    boundary_points = cavity_np[convex_hull.vertices]

    # compute distances between center and boundary points
    distances = np.linalg.norm(boundary_points - center, axis=1)

    # find the indices of the furthest and closest point
    furthest_point_index = np.argmax(distances)
    closest_point_index = np.argmin(distances)

    # get coordinates of the furthest and closest point
    furthest_point = boundary_points[furthest_point_index]
    closest_point = boundary_points[closest_point_index]

    # calculate distance to the furthest and closest points
    dist_to_furthest_point = distances[furthest_point_index]
    dist_to_closest_point = distances[closest_point_index]

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
    angle_deg = np.degrees(angle_radians)

    return dist_to_closest_point, dist_to_furthest_point, angle_deg


def find_neighboring_residues(prot_file, cav_file, distance_threshold=4.0):
    """
    Identify and retrieve the residue indices of atoms within a specified distance threshold from a cavity within a
    protein structure.

    :param prot_file: str, The file path to the protein structure in a format compatible with PyMOL.
    :param cav_file: str, The file path to the cavity structure in a format compatible with PyMOL.
    :param distance_threshold: float, optional, The distance threshold (in angstroms) used to filter atoms within the
        cavity. Defaults to 4.0 angstroms.

    :return: set, A set containing the residue indices of atoms within the specified distance threshold from the cavity.
    """
    # Load ligand and protein in PyMOL
    cmd.load(prot_file)
    cmd.load(cav_file)

    cavity_obj = cav_file.split('/')[-1].split('.')[0]

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

    cmd.reinitialize()

    return resid_indices


def is_residue_exposed_to_cavity(protein, cavity, residue_id):
    """
    Determine whether a residue is exposed to a cavity in a protein structure based on the cosine of angles.

    :param protein: DataFrame, The protein structure data containing information about atoms.
    :param cavity: DataFrame, The cavity structure data containing information about atoms.
    :param residue_id: int, The identifier of the residue to be checked for exposure.

    :return: tuple (bool, str or None), A tuple indicating whether the residue is exposed and, if so, whether it is the
        'side_chain' or 'backbone'.
    """
    cavity_points = get_points(cavity)
    cavity_center = center_of_gravity(cavity_points)

    residue = protein[protein['subst_id'] == residue_id]

    # Calculate the vector between the residue's backbone (N, CA, C) and the cavity's center of gravity
    backbone_atoms = ['N', 'CA', 'C', 'O']
    backbone = pd.concat([residue[residue['atom_name'] == atom] for atom in backbone_atoms], ignore_index=True)
    backbone_center = center_of_gravity(get_points(backbone))

    # Get coordinates of CA of the residue
    CA = residue[residue['atom_name'] == 'CA'][["x", "y", "z"]].values
    CA_coords = pd.Series(CA[0], index=["x", "y", "z"])

    # Calculate the vector between the residue's side chain and the cavity's center of gravity
    side_chain_atoms = np.setdiff1d(np.unique(protein[['atom_name']].values), backbone_atoms)
    side_chain = pd.concat([residue[residue['atom_name'] == atom] for atom in side_chain_atoms], ignore_index=True)
    side_chain_center = center_of_gravity(get_points(side_chain))

    backbone_side_chain_vector = np.array(side_chain_center - CA_coords)
    residue_cavity_vector = np.array(cavity_center - CA_coords)
    cosine_angle = np.dot(backbone_side_chain_vector, residue_cavity_vector) / (
            np.linalg.norm(backbone_side_chain_vector) * np.linalg.norm(residue_cavity_vector))

    # Add min & max distance to the center of the gravity & shell & the angle between those
    convex_hull = convexhull(cavity_points)
    dist_to_closest_point, dist_to_furthest_point, angle_deg = (
        distances_angles_shell_center(cavity_points, convex_hull))

    backbone_cavity_dist = np.linalg.norm(backbone_center - cavity_center)

    sphere_dist = math.sqrt(dist_to_furthest_point**2 + backbone_cavity_dist**2)

    threshold = (backbone_cavity_dist**2 + sphere_dist**2 - dist_to_furthest_point**2) / (
            2 * backbone_cavity_dist * sphere_dist)

    if threshold <= cosine_angle <= 1:
        return True, 'side_chain'
    elif -1 <= cosine_angle <= -threshold:
        return True, 'backbone'
    else:
        return False, None


def get_exposed_residues(prot_file, protein, cav_file, cavity, distance_threshold=4.0):
    """
    Analyze and classify exposed residues surrounding a cavity in a protein structure.

    :param prot_file: str, The file path to the protein structure in a format compatible with PyMOL.
    :param protein: DataFrame, The protein structure data containing information about atoms.
    :param cav_file: str, The file path to the cavity structure in a format compatible with PyMOL.
    :param cavity: DataFrame, The cavity structure data containing information about atoms.
    :param distance_threshold: float, optional, The distance threshold (in angstroms) for identifying neighboring
        residues. Defaults to 4.0 angstroms.

    :return: DataFrame, A DataFrame containing information about exposed residues.
    """
    # Get the residues that surround the cavity
    neighboring_residues = find_neighboring_residues(prot_file, cav_file, distance_threshold=distance_threshold)

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
            resn = protein['subst_name'][protein['subst_id'] == residue].values[0]
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
                        'exposed_backbone_ratio_all': float(exposed_backbone / all_exposed) if all_exposed > 0 else 0.0,
                        'exposed_side_chain_ratio_all': float(exposed_side_chain /
                                                              all_exposed) if all_exposed > 0 else 0.0,
                        'exposed_polar_side_ratio': float(polar_side_chain /
                                                          exposed_side_chain) if exposed_side_chain > 0 else 0.0,
                        'exposed_aromatic_side_ratio': float(aromatic_side_chain /
                                                             exposed_side_chain) if exposed_side_chain > 0 else 0.0,
                        'exposed_pos_side_ratio': float(pos_side_chain /
                                                        exposed_side_chain) if exposed_side_chain > 0 else 0.0,
                        'exposed_neg_side_ratio': float(neg_side_chain /
                                                        exposed_side_chain) if exposed_side_chain > 0 else 0.0,
                        'exposed_hydrophobic_side_ratio': float(hydrophobic_side_chain /
                                                                exposed_side_chain) if exposed_side_chain > 0 else 0.0
                        }

    df_exposed = pd.DataFrame(exposed_residues, index=[0])
    return df_exposed


def sphericity(cavity_points):
    # Calculate the convex hull
    convex_hull = ConvexHull(cavity_points)

    # Calculate the volume of the convex hull
    hull_volume = convex_hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = convex_hull.area

    # Calculate the radius of a sphere with the same volume
    sphere_radius = ((3 * hull_volume) / (4 * np.pi))**(1/3)

    # Calculate the surface area of a sphere with the same volume
    sphere_surface_area = 4 * np.pi * (sphere_radius**2)

    # Calculate the sphericity
    sphericity_ratio = hull_surface_area / sphere_surface_area

    return sphericity_ratio


def cubic_sphericity(cavity_points):
    # Calculate the convex hull
    convex_hull = ConvexHull(cavity_points)

    # Calculate the volume of the convex hull
    hull_volume = convex_hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = convex_hull.area

    # Calculate the side length of a cube with the same volume
    cube_side_length = hull_volume ** (1 / 3)

    # Calculate the surface area of a cube with the same volume
    cube_surface_area = 6 * (cube_side_length**2)

    # Calculate the cubic sphericity
    cubic_sphericity_ratio = hull_surface_area / cube_surface_area

    return cubic_sphericity_ratio


def cone_sphericity(cavity_points):
    # Calculate the convex hull
    convex_hull = ConvexHull(cavity_points)

    # Calculate the volume of the convex hull
    hull_volume = convex_hull.volume

    # Calculate the surface area of the convex hull
    hull_surface_area = convex_hull.area

    # Calculate the radius and height of a cone with the same volume
    cone_radius = np.sqrt(hull_surface_area / (np.pi * hull_volume))
    cone_height = hull_volume / (np.pi * cone_radius**2)

    # Calculate the surface area of a cone with the same volume
    cone_surface_area = np.pi * cone_radius * (cone_radius + np.sqrt(cone_radius**2 + cone_height**2))

    # Calculate the cone sphericity
    cone_sphericity_ratio = hull_surface_area / cone_surface_area

    return cone_sphericity_ratio


def find_obb(cavity_points):
    means = cavity_points.mean()
    points = cavity_points - means
    # Apply PCA to find principal components and directions
    pca = PCA(n_components=3)
    pca.fit(points)

    # The components_ attribute contains the principal axes
    principal_axes = pca.components_

    # Calculate the extent along each principal axis
    extent = np.max(points.dot(principal_axes.T), axis=0) - np.min(points.dot(principal_axes.T), axis=0)
    return extent


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
    # Usage:
    # python3 main.py [volsite_output_folder] [descriptor_csv_file]
    volsite_output = argv[1]
    input_proteins = list_subdirectories(volsite_output)

    output_csv = argv[2]
    if not output_csv.endswith('.csv'):
        output_csv = argv[2] + '.csv'
    all_descriptors = pd.DataFrame()

    for protein_volsite in input_proteins:
        protein_code = protein_volsite.split('/')[-1]
        print(f'Calculating descriptors for {protein_code}...')

        protein_path = f'{protein_volsite}/protein_no_solvent.mol2'
        protein_df = load_mol_file(protein_path)

        ligand_path = f'{protein_volsite}/ligand.mol2'

        # Select the cavity that covers the ligand
        cavity_file, cavity_index, cavity_df = select_cavity(protein_volsite, ligand_path)
        cavity_path = f'{protein_volsite}/{cavity_file}'

        if cavity_df is not None and load_mol_file(protein_path) is not None:
            print(f'Cavity size: {cavity_df.shape[0]}')

            # Get descriptors generated by Volsite
            print('Retrieving descriptors calculated by Volsite...', end=' ')
            volsite_descriptors = get_volsite_descriptors(protein_volsite, cavity_index)
            cavity_descriptors = pd.DataFrame({'protein_code': [protein_code]})
            cavity_descriptors = pd.concat([cavity_descriptors, pd.DataFrame(volsite_descriptors).transpose()], axis=1)
            cavity_descriptors.reset_index(drop=True, inplace=True)
            print('Done')

            # Cavity X, Y, Z coordinates
            cavity_points_df = get_points(cavity_df)
            # Add cavity area to the df
            print('Calculating cavity area...', end=' ')
            hull = convexhull(cavity_points_df)
            cavity_area = area(hull)
            cavity_descriptors = cavity_descriptors.assign(area=[cavity_area])
            print('Done')

            # Add min & max distance to the center of the gravity & shell & the angle between those
            print('Calculating min & max distance from the center of the gravity the shell & the angle between '
                  'those...', end=' ')
            distance_to_closest_point, distance_to_furthest_point, angle_degrees = (
                distances_angles_shell_center(cavity_points_df, hull))
            cavity_descriptors = cavity_descriptors.assign(min_dist=[distance_to_closest_point])
            cavity_descriptors = cavity_descriptors.assign(max_dist=[distance_to_furthest_point])
            cavity_descriptors = cavity_descriptors.assign(angle=[angle_degrees])
            print('Done')

            # Add max dist between two cavity points to the descriptors df
            print('Calculating max distance between pharmacophore points...', end=' ')
            pharmacophore_df = load_mol_file(f'{protein_volsite}/Pharmacophore.mol2')
            max_dist_pairs = max_dist_cavity_points(pharmacophore_df)
            cavity_descriptors = pd.concat([cavity_descriptors, max_dist_pairs], axis=1)
            print('Done')

            # Get residues exposed to the cavity
            print('Calculating residues exposed to the cavity...', end=' ')
            exposed_aa = get_exposed_residues(protein_path, protein_df, cavity_path, cavity_df)
            cavity_descriptors = pd.concat([cavity_descriptors, exposed_aa], axis=1)
            print('Done')

            # Calculate the sphere shape ration descriptors
            print('Calculating the sphere-like ratio...', end=' ')
            sphere = sphericity(cavity_points_df)
            # Add shape ration descriptors to the df
            cavity_descriptors = cavity_descriptors.assign(sphere=[sphere])
            print('Done')

            # Compute the smallest box
            print('Calculating the smallest box...', end=' ')
            smallest_box = find_obb(cavity_points_df)
            cavity_descriptors = cavity_descriptors.assign(box_x=[smallest_box.iloc[0]])
            cavity_descriptors = cavity_descriptors.assign(box_y=[smallest_box.iloc[1]])
            cavity_descriptors = cavity_descriptors.assign(box_z=[smallest_box.iloc[2]])
            print('Done')

            # Make the plot and save it
            print('Saving a plot of the cavity...', end=' ')
            save_path = f'{protein_volsite}/cavity_plot.png'
            plot_cavity(cavity_points_df.to_numpy(), hull, save_path)
            print('Done')

            # Add the descriptors of the current cavity to the general dataframe
            if all_descriptors.empty:
                all_descriptors = cavity_descriptors
            else:
                all_descriptors = pd.concat([all_descriptors, cavity_descriptors.iloc[[-1]]], ignore_index=True)

        else:
            print('No cavity for this structure')
            no_cavity = pd.DataFrame()
            no_cavity = no_cavity.set_axis([0], axis=0)
            no_cavity.insert(0, "protein_code", protein_code)
            all_descriptors = pd.concat([all_descriptors, no_cavity.iloc[[-1]]], ignore_index=True)

        print(f'{protein_code} done')
        print('-----------------------------------')
        all_descriptors.to_csv(output_csv)

    all_descriptors.to_csv(output_csv)
    print('Calculation of descriptors for each structure is finished!')
