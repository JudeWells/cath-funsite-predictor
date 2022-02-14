import pickle
import pandas as pd

directory = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/experiments/pdbs/'
with open(directory + 'atom_groups.pickle', 'rb') as handle:
    atom_groups = pickle.load(handle)

count_resnames = {}
for k in atom_groups.keys():
    for r in atom_groups[k].getResnames():
        count_resnames[r] = count_resnames.get(r,0) + 1

# MSE is somewhat common (Selenomethionine)
valid_residues = ['LEU', 'GLU', 'ARG', 'VAL', 'LYS', 'ILE', 'ASP', 'PHE', 'ALA', 'TYR', 'THR', 'SER', 'GLN', 'PRO', 'ASN', 'GLY', 'HIS', 'MET', 'TRP', 'CYS']

breakpoint_var = True


