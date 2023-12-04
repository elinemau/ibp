import pandas as pd
from pymol import cmd
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

    #extract all information about the specific residue in the protein dataframe
    residue = protein[protein['subst_name'].str.endswith(str(residue_id))]

    # Calculate the vector between the residue's backbone (N, CA, C) and the cavity's center of gravity
    backbone_atoms = ['N', 'CA', 'C', 'O']
    backbone = pd.concat([residue[residue['atom_name'] == atom] for atom in backbone_atoms], ignore_index=True)
    backbone_center = center_of_gravity(get_points(backbone))
    #vector starts from backbone
    backbone_direction_vector = np.array(cavity_center - backbone_center)

    # Calculate the vector between the residue's side chain and the cavity's center of gravity
    side_chain_atoms = np.setdiff1d(np.unique(protein[['atom_name']].values), backbone_atoms)
    side_chain = pd.concat([residue[residue['atom_name'] == atom] for atom in side_chain_atoms], ignore_index=True)
    side_chain_center = center_of_gravity(get_points(side_chain))
    side_chain_direction_vector = np.array(cavity_center - side_chain_center)


    # visualization of vectors and points
    #get protein and cavity point cloud
    protein_points = get_points(protein).to_numpy()
    protein_hull = ConvexHull(protein_points)
    cavity_points = get_points(cavity).to_numpy()
    cavity_hull = ConvexHull(cavity_points)
    #get residue points
    residue_points = residue[["x","y","z"]].to_numpy()
    # make fig to plot  mesh
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cavity_mesh = Poly3DCollection([cavity_points[s] for s in cavity_hull.simplices], alpha=0.25, edgecolor='k')
    ax.add_collection3d(cavity_mesh)
    """protein_mesh = Poly3DCollection([protein_points[s] for s in protein_hull.simplices], alpha=0.25, edgecolor='k')
    ax.add_collection3d(protein_mesh)"""
    ax.scatter(residue_points[:, 0], residue_points[:, 1], residue_points[:, 2], c='r', marker='o',
               label='residue points')
    #ax.scatter(cavity_points[:, 0], cavity_points[:, 1], cavity_points[:, 2], c='b', marker='o', label='cavity')
    ax.quiver(backbone_center[0], backbone_center[1], backbone_center[2], backbone_direction_vector[0], backbone_direction_vector[1], backbone_direction_vector[2], label='backbone_direction_vector')
    ax.quiver(side_chain_center[0], side_chain_center[1], side_chain_center[2],side_chain_direction_vector[0], side_chain_direction_vector[1], side_chain_direction_vector[2])
    ax.set_xlim([18, 28])
    ax.set_ylim([5, 20])
    ax.set_zlim([45,65])
    plt.show()

    # Calculate the cosine of the angle between vectors
    backbone_cosine_angle = np.dot(backbone_direction_vector, cavity_center) / (
            np.linalg.norm(backbone_direction_vector) * np.linalg.norm(cavity_center))
    print(backbone_cosine_angle)
    print(backbone_direction_vector)
    side_chain_cosine_angle = np.dot(side_chain_direction_vector, cavity_center) / (
            np.linalg.norm(side_chain_direction_vector) * np.linalg.norm(cavity_center))
    print(side_chain_cosine_angle)

    # Check if the cosine of the angles are greater than the threshold to determine exposure
    if side_chain_cosine_angle > dot_product_threshold:
        return True, 'side_chain'
    elif backbone_cosine_angle > dot_product_threshold:
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