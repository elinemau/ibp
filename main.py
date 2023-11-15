import pandas as pd
import numpy as np
from itertools import combinations_with_replacement
from sklearn.decomposition import PCA
import math
import os


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
            data[-1] = map(float, data[-1])
            df.loc[len(df)] = data
            line = file.readline()
    file.close()
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['z'] = df['z'].astype(float)
    return df

def center_of_gravity(df):
    """

    :param df: df generated from file with def load_mol_file
    :return: center of gravity for the given structure
    """
    # Calculate the center of gravity of the structure
    COG = df[["x", "y", "z"]].mean()
    COG = COG.to_frame()
    return COG


def calculate_nearest_point(df, reference):
    """

    :param df: columns are point, row 1=x, row 2=y, row3=z
    :param reference: the reference point with same structure as df
    :return: the index of the column in df that is closest to reference
    """
    distance = []
    for column in df:
        d = math.sqrt((reference[0].values[0]-df.values[0][column])**2 + (reference[0].values[1]-df.values[1][column])**2 + (reference[0].values[2]-df.values[2][column])**2)
        distance.append(d)
    return distance.index(min(distance))


def select_cavity(folder):
    """
    Investigates all cavities found with Volsite (with ligand restriction) and selects the one closest
    to the center of gravity of the ligand.

    :return: df of the cavity closest to the ligand
    """
    cog = pd.DataFrame()
    files = []
    # select cavity files from the folder and put them in a list
    for file in os.listdir(folder):
        # include ALL in name, because N2, N4, n6,... are dublicate files
        if "CAVITY" and "ALL" in file:
            f = os.path.join(folder, file)
            # get df from file
            df = load_mol_file(f)
            # calculate center of gravity
            center = center_of_gravity(df)
            cog = pd.concat([cog, center], axis=1)
            files.append(str(file))
        elif "protein" in file:
            f = os.path.join(folder, file)
            # get df from file
            df = load_mol_file(f)
            # calculate center of gravity
            protein_center = center_of_gravity(df)
        else:
            continue
    # compare the distance from the cavities to the ligand and return the closest cavity
    index = calculate_nearest_point(cog, protein_center)
    return files[index]


def get_volsite_descriptors(filename, cavity_num):
    """
    Loads the descriptors for a given cavity returned by Volsite into a pandas dataframe.

    :param filename: file name with volsite descriptors (.txt)
    :param cavity_num: number of the selected cavity (the one closer to the ligand)
    :return: dataframe with volsite descriptors for a given cavity
    """
    # add a failsafe for the incorrect cavity_num later!!!
    points = ['CZ', 'CA', 'O', 'OD1', 'OG', 'N', 'NZ', 'DU']
    column_names = ['volume'] + points
    for point in points:
        column_names += [f'{point}_below_40', f'{point}_between_40_50', f'{point}_between_50_60', f'{point}_between_60_70',
                         f'{point}_between_70_80', f'{point}_between_80_90', f'{point}_between_90_100',
                         f'{point}_between_100_110', f'{point}_between_110_120', f'{point}_120']
    column_names += ['name']
    df = pd.read_csv(filename, sep=" ", index_col=False, header=None, names=column_names)
    return df.loc[cavity_num-1, df.columns != 'name']


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

def pc_retrieval(df):
    """

    :param df: pandas dataframe output from load_mol_file
    :return: first and second principal component
    """
    point_cloud = df[["x", "y", "z"]].to_numpy()
    pca=PCA(n_components=3)
    pca.fit(point_cloud)
    return pca.components_

if __name__ == '__main__':
    cavity = select_cavity("C:\\Users\\32496\\PycharmProjects\\IBP\\1a28\\volsite")
    print(cavity)
    # cavity = load_mol_file("../1a28/volsite/CAVITY_N1_ALL.mol2")
    # volsite_descriptors = get_volsite_descriptors("../1a28/volsite/1a28_prot_no_waters_descriptor.txt", 1)
    # print(volsite_descriptors)
    # print(max_dist_cavity_points(cavity))
