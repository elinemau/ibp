import pandas as pd
import numpy as np
from itertools import combinations, combinations_with_replacement, product
import math


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


def center_of_gravity(filename):
    """

    :param filename: file name of the structure (mol2 format)
    :return: center of gravity for the given structure
    """
    # Load the mol file (with the load_mol_file function)
    atoms = load_mol_file(filename)
    # Calculate the center of gravity of the structure
    COG = atoms[["x", "y", "z"]].mean()
    return COG


def select_cavity():
    """
    Investigates all cavities found with Volsite (with ligand restriction) and selects the one closest
    to the center of gravity of the ligand.

    :return: df of the cavity closest to the ligand
    """
    # compare the distance from the cavities to the ligand and return the closest cavity

    pass


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


def distance(point1, point2):
    """
    Calculates the distance between two points in 3D space.

    :param point1: first point in 3D space
    :param point2: second point in 3D space
    :return: distance between point1 and point2
    """
    return math.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(point1, point2)))


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


if __name__ == '__main__':
    cavity = load_mol_file("../1a28/volsite/CAVITY_N1_ALL.mol2")
    volsite_descriptors = get_volsite_descriptors("../1a28/volsite/1a28_prot_no_waters_descriptor.txt", 1)
    # print(volsite_descriptors)
    print(max_dist_cavity_points(cavity).iloc[:, :5])
    print(max_triplet_area(cavity))
