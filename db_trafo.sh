#!/bin/bash

#before running the code, 
# - the cd should be a directory containing the Iridium dataset (original format)
# - the cd should contain the pymol script 'extract_mol2.py'

#directory setup-------------------------------------------------------------------------------------------
mkdir -p 021no_water
mkdir -p 022pymol_conv
mkdir -p 023volsite_desc_output

#deletion of H2O mols--------------------------------------------------------------------------------------
FILES=Iridium/iridium-HT/deposited/*_deposited_refined_prot.pdb
for f in $FILES
do
   #read the protein ID of the file (by removing suffix)
   f_front=${f%_deposited_refined_prot.pdb}
   f_id=${f_front#Iridium/iridium-HT/deposited/}

   #delete the water molecules from the structure
   grep 'HOH' -v $f > 021no_water/${f_id}_noH2O.pdb
done


#save as mol2 and run volsite------------------------------------------------------------------------------
#select one chain of structure (to analyze ONLY one binding site) and save as .mol2 
#in the same forloop we run the volsite program

#define the PyScript we'll use
script=save_mol2.py

#execute PyScript for every structure
for f in 021no_water/*_noH2O.pdb
do 
   #read the protein ID of the file (by removing suffix)
   f_front=${f%_noH2O.pdb}
   f_id=${f_front#021no_water/} 

   #generate mol2 outfile name
   mol2_outfile=022pymol_conv/${f_id}

   #run the script
   /opt/pymol/pymol -c $script -- $f ${mol2_outfile}.mol2
   echo "Created ${f_id}.mol2"

   #save the [ligand].sdf file as a .mol2
   /opt/pymol/pymol -c $script -- Iridium/iridium-HT/deposited/${f_id}_deposited_refined_lig.sdf ${mol2_outfile}_lig.mol2
   echo "Created ${f_id}_lig.mol2"
   #the conversion to .mol2 has a faulty conversion of some atoms
   grep -rl 'S.O2' ${mol2_outfile}_lig.mol2 | xargs sed -i 's/S.O2/S.2/g' 
   grep -rl 'S.O' ${mol2_outfile}_lig.mol2 | xargs sed -i 's/S.O/S/g' 
   

   #run volsite to output descriptor and pharmacophore files
   mkdir -p 023volsite_desc_output/${f_id}
   cd 023volsite_desc_output/${f_id}
   /opt/IChem_files/IChem --desc volsite ../../${mol2_outfile}.mol2 ../../${mol2_outfile}_lig.mol2
   /opt/IChem_files/IChem --desc --pharm volsite ../../${mol2_outfile}.mol2 ../../${mol2_outfile}_lig.mol2
   cd ../..
done

