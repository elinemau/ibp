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
    return df


def center_of_gravity(filename):
    """

    :param filename: file name of the structure (mol2 format)
    :return: center of gravity for the given structure
    """
    # Load the mol file (with the load_mol_file function)
    atoms = load_mol_file(filename)
    # Calculate the center of gravity of the structure
    pass


def select_cavity():
    """
    Investigates all cavities found with Volsite (with ligand restriction) and selects the one closest
    to the center of gravity of the ligand.

    :return: df of the cavity closest to the ligand
    """
    # compare the distance from the cavities to the ligand and return the closest cavity
    pass


if __name__ == '__main__':
    cavity = load_mol_file("1a28/volsite/CAVITY_N1_ALL.mol2")
    print(cavity)
