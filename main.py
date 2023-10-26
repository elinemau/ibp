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


if __name__ == '__main__':
    cavity = load_mol_file("volsite/CAVITY_N1_ALL.mol2")
