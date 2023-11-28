import sys   

#read input and output filenames from script arguments
infile = sys.argv[1]
mol2_outfile = sys.argv[2]

cmd.load(infile) #load struc
cmd.select("struc", "all")
cmd.save(mol2_outfile, "struc") #save as .mol2 file
