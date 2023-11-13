import pandas as pd


def load_mol_file(filename):
    df = pd.DataFrame(columns=['atom_id', 'atom_name', 'x', 'y', 'z', 'atom_type', 'subst_id', 'subst_name', 'charge'])
    with open(filename, "r") as file:
        line = file.readline()
        while not line.startswith("@<TRIPOS>ATOM"):
            line = file.readline()
        line = file.readline()
        while not line.startswith("@<TRIPOS>BOND"):
            df.loc[len(df)] = line.strip().split()
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


if __name__ == '__main__':
    cavity = load_mol_file("../1a28/volsite/CAVITY_N1_ALL.mol2")
    volsite_descriptors = get_volsite_descriptors("../1a28/volsite/1a28_prot_no_waters_descriptor.txt", 1)
    print(volsite_descriptors)
