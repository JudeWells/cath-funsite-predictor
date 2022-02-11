import numpy as np
import pandas as pd
import pickle
import os
import prody


def load_rep(dom_id):
    """
    This function takes a domain id and returns the corresponding alphafold
    representation for this domain as a numpy.ndarray
    """
    upper_dom_id = dom_id.upper()
    directory = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/alpha_pickles/representation_pickles/'
    try:
        with open(directory + upper_dom_id + '.pickle', 'rb') as handle:
            representation = pickle.load(handle)
        return representation
    except:
        pass

def add_rep_to_res(df, domain, representation, alignment):
    """
    Updates the dataframe to include the alphafold representation as a string
    alignment is a dictionary that maps from the domain_residue index to the
    corresponding index in the alphafold structure.
    """
    one_domain = df[df.domain == domain]
    for i, row in one_domain.iterrows():
        residue_idx = row.domain_residue
        alphafold_idx = alignment[residue_idx]
        df.loc[i, 'alpha_embed'] = str(representation['single'][residue_idx, :])

def get_all_atom_groups(df, output_dir='pdbs', pickle_grps=True):
    atom_groups = {}
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)
    for dom_str in df.domain.unique():
        pdb_id = dom_str[:-3]
        chain = dom_str[-3].upper()
        atom_grp = prody.parsePDB(pdb_id, chain=chain)
        atom_groups[dom_str] = atom_grp

    with open('atom_groups.pickle', 'wb') as handle:
        pickle.dump(atom_groups, handle)
    return atom_groups

def make_alignment(domain_id):
    """
    Think about what must be true for this to work:
    we have a list of numbers representing the residue number for each atom in the protein
    Do the residue numbers have gaps in them if the residue is missing from the structure?
    """
    atom_group = atom_groups[domain_id]
    resnum = atom_group.getData('resnum') # returns a list of numbers that correspond to the indexing used to label the PPI dataset
    alignment = {}
    return alignment

if __name__=="__main__":
    df = pd.read_csv('../datasets/PPI/PPI_training_dataset.csv')
    atom_groups = get_all_atom_groups(df.iloc[:50,])
    breakpoint_var = True



