from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors

# Load the Mol2 files for protein and ligand
protein_mol2_file = '1a28/volsite/protein.mol2'
ligand_mol2_file = '1a28/volsite/ligand.mol2'

protein_mol = Chem.MolFromMol2File(protein_mol2_file)
ligand_mol = Chem.MolFromMol2File(ligand_mol2_file)

# Identify chemical bonds
AllChem.AssignBondOrdersFromTemplate(protein_mol, ligand_mol)

# Calculate intermolecular interactions
num_hbd_protein = rdMolDescriptors.CalcNumHBD(protein_mol)
num_hba_ligand = rdMolDescriptors.CalcNumHBA(ligand_mol)

# Visualize the protein and ligand (you can customize this part)
img = Draw.MolToImage(protein_mol)
img.save('protein.png')
img = Draw.MolToImage(ligand_mol)
img.save('ligand.png')

print(f'Number of hydrogen bond donors in protein: {num_hbd_protein}')
print(f'Number of hydrogen bond acceptors in ligand: {num_hba_ligand}')
