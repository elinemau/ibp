import pandas as pd
from pymol import cmd
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math

def get_points(df):
    """

    :param df: mol2 file processed by load_mol2_file
    :return: coordinates
    """
    return df[["x", "y", "z"]]


def load_mol_file(filename):
    """
    Loads the descriptors for a given cavity returned by Volsite into a pandas DataFrame.

    :param filename: file name with Volsite descriptors (.txt).
    :param cavity_index: int: Index of the selected cavity (the one closer to the ligand).
    :return: pandas.DataFrame or None: DataFrame with Volsite descriptors for a given cavity, returns None if unsuccessful.
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


def center_of_gravity(points):
    """

    :param points: coordinates retreived from get_points
    :return: center of gravity for the given structure
    """
    # Calculate the center of gravity of the structure
    return points.mean()


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

    #next code is to see the model, we output it to an excel file
    """data = {'Atom': [], 'Residue': [], 'Chain': [], 'X': [], 'Y': [], 'Z': []}
    # Extract atom information from the model and populate the DataFrame
    for atom in model.atom:
        data['Atom'].append(atom.name)
        data['Residue'].append(atom.resi)
        data['Chain'].append(atom.chain)
        data['X'].append(atom.coord[0])
        data['Y'].append(atom.coord[1])
        data['Z'].append(atom.coord[2])

    df= pd.DataFrame(data)
    df.to_excel('output.xlsx', index=False)"""

    #back to original code
    #residues are the aminoacids, every atom belongs to a residue and a residue has multiple atoms
    res_lim = model.get_residues()

    atom_list = model.atom
    resid_indices = set()

    for start, end in res_lim:  # extract the data we are interested in
        #with start, end, we can select all atoms in one residue. start is the startatom of the residue and end is the end atom of that residue
        for atom in atom_list[start:end]:
            resid_indices.add(atom.resi)

    #returns set of residues present around cavity
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

    print(residue_id)
    residue = protein[protein['subst_id'] == residue_id]
    # residue_center = center_of_gravity(get_points(residue))

    # Calculate the vector between the residue's backbone (N, CA, C) and the cavity's center of gravity
    backbone_atoms = ['N', 'CA', 'C', 'O']
    backbone = pd.concat([residue[residue['atom_name'] == atom] for atom in backbone_atoms], ignore_index=True)
    backbone_center = center_of_gravity(get_points(backbone))
    # backbone_direction_vector = np.array(cavity_center - backbone_center)

    # Calculate the vector between the residue's side chain and the cavity's center of gravity
    side_chain_atoms = np.setdiff1d(np.unique(protein[['atom_name']].values), backbone_atoms)
    side_chain = pd.concat([residue[residue['atom_name'] == atom] for atom in side_chain_atoms], ignore_index=True)
    side_chain_center = center_of_gravity(get_points(side_chain))
    # side_chain_direction_vector = np.array(cavity_center - side_chain_center)

    backbone_side_chain_vector = np.array(side_chain_center - backbone_center)
    residue_cavity_vector = np.array(cavity_center - backbone_center)
    cosine_angle = np.dot(backbone_side_chain_vector, residue_cavity_vector) / (
            np.linalg.norm(backbone_side_chain_vector) * np.linalg.norm(residue_cavity_vector))
    print(f'cosine_angle: {cosine_angle}')
    cavity_hull = ConvexHull(cavity_points)
    distance_to_closest_point, cavity_radius, angle_degrees = (distances_angles_shell_center(cavity_points,cavity_hull))

    backbone_cavity_dist = np.linalg.norm(backbone_center - cavity_center)
    print(f'backbone_cavity_dist: {backbone_cavity_dist}')

    sphere_dist = math.sqrt(cavity_radius**2 + backbone_cavity_dist**2)
    print(f'sphere_dist: {sphere_dist}')

    threshold = (backbone_cavity_dist**2 + sphere_dist**2 - cavity_radius**2) / (
            2 * backbone_cavity_dist * sphere_dist)
    print(f'threshold: {threshold}')

    # get residue points
    residue_points = residue[["x", "y", "z"]].to_numpy()
    # make fig to plot  mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cavity_mesh = Poly3DCollection([cavity_points[s] for s in cavity_hull.simplices], alpha=0.25, edgecolor='k')
    ax.add_collection3d(cavity_mesh)
    ax.scatter(residue_points[:, 0], residue_points[:, 1], residue_points[:, 2], c='r', marker='o',
               label='residue points')
    ax.quiver(backbone_center[0], backbone_center[1], backbone_center[2], backbone_direction_vector[0],
              backbone_direction_vector[1], backbone_direction_vector[2], label='backbone_direction_vector')
    ax.quiver(side_chain_center[0], side_chain_center[1], side_chain_center[2], side_chain_direction_vector[0],
              side_chain_direction_vector[1], side_chain_direction_vector[2])
    ax.set_xlim([18, 28])
    ax.set_ylim([5, 20])
    ax.set_zlim([45, 65])
    plt.show()

    if threshold <= cosine_angle <= 1:
        print("side chain")
        return True, 'side_chain'
    elif -1 <= cosine_angle <= -threshold:
        print("backbone")
        return True, 'backbone'
    else:
        return False, None


protein_file = "1a28\\protein.mol2"
cavity_file = "1a28/volsite/CAVITY_N1_ALL.mol2"
residues = find_neighboring_residues(protein_file, cavity_file)
protein_df = load_mol_file(protein_file)
cavity_df = load_mol_file(cavity_file)
for i in residues:
    if i == "894":
        print(is_residue_exposed_to_cavity(protein_df, cavity_df,i , dot_product_threshold=0.0))
    else:
        continue
#'887', '715', '890', '891', '801', '909', '903', '722', '719', '894', '718', '725', '905', '794', '778', '721', '797', '760', '766', '759', '763', '755', '756'