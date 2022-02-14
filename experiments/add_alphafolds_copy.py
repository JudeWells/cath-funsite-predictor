import re
import gzip
import numpy as np
import pandas as pd
import pickle
import os
import prody
from prody.atomic.atomic import Atomic

valid_residues = ['LEU', 'GLU', 'ARG', 'VAL', 'LYS', 'ILE', 'ASP', 'PHE', 'ALA', 'TYR', 'THR', 'SER', 'GLN', 'PRO',
                  'ASN', 'GLY', 'HIS', 'MET', 'TRP', 'CYS']

valid_res_letters = 'GAVCPLIMWFKRHSTYNQDE'

def load_atom_groups_from_pickle():
    dir = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/experiments/pdbs/'
    with open(dir + 'atom_groups.pickle', 'rb') as handle:
        atom_groups = pickle.load(handle)
    return atom_groups

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

def find_expected_true_length(atom_group):
    valid_residues = ['LEU', 'GLU', 'ARG', 'VAL', 'LYS', 'ILE', 'ASP', 'PHE', 'ALA', 'TYR', 'THR', 'SER', 'GLN', 'PRO',
                      'ASN', 'GLY', 'HIS', 'MET', 'TRP', 'CYS']
    num_2_res = dict(zip(atom_group.getData('resnum'), atom_group.getResnames()))
    refined_num_2_res = {k:v for k,v in num_2_res.items() if v in valid_residues}
    expected_residues = 1 + max(refined_num_2_res.keys()) - min(refined_num_2_res.keys())
    return expected_residues

def check_true_length():
    error_counter = 0
    match_counter = 0
    for i, row in df.iterrows():
        domid = re.split('[0-9]+_',row.residue_string)[0].upper()

        representation = load_rep(domid)
        if representation is None:
            continue
        rep_shape = representation['single'].shape[0]
        if rep_shape == find_expected_true_length(atom_groups[row.domain]):
            match_counter += 1
        else:
            error_counter += 1
    print(f'error: {error_counter}, match: {match_counter}')


def gzip_open(filename, *args, **kwargs):
    if args and "t" in args[0]:
        args = (args[0].replace("t", ""), ) + args[1:]
    if isinstance(filename, str):
        return gzip.open(filename, *args, **kwargs)
    else:
        return gzip.GzipFile(filename, *args, **kwargs)


def load_pdb_lines(pdb_code):
    dir = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/experiments/pdbs/'
    filepath = dir + pdb_code.lower() + '.pdb.gz'
    with gzip_open(filepath) as handle:
        lines = handle.readlines()
    str_lines = [str(l) for l in lines]
    return str_lines


def make_alignment(domain_id):
    """
    Think about what must be true for this to work:
    we have a list of numbers representing the residue number for each atom in the protein
    The residue numbers have gaps where the missing numbers are not included.
    In some cases there are non-amino acid residue names that are also included in the count and this can mess the numbers up.
    atom_groups[row.domain]._getSN2I() - this returns an array that has some -1s in it. It seems to be the inverse of:
    atom_groups[row.domain]._data['serial']
    atom_groups[row.domain].numResidues()

    [l for l in lines if 'SEQRES' in l] # this will return all the lines that list out the sequence information
    [l for l in lines if 'REMARK 465' in l] # this will return each line that describes a missing residue

    A good place to add missing residue functionality would be in this function:
    _parsePDBLines() prody/proteins/pdbfiles
    """
    atom_group = atom_groups[domain_id]
    resnum = atom_group.getData('resnum') # returns a list of numbers that correspond to the indexing used to label the PPI dataset
    alignment = {}
    return alignment

def flatten(nested):
    return [item for sublist in nested for item in sublist]

def get_sequence_from_lines(lines, chain):
    sequence_lines = [l for l in lines if 'SEQRES' in l]
    one_chain = [l.split()[4:] for l in sequence_lines if l.split()[2]==chain]
    one_chain_flat = flatten(one_chain)
    one_chain_flat = [resname for resname in one_chain_flat if resname in valid_residues]
    return one_chain_flat

def get_missing_from_lines(lines, chain):
    missing_dict = {}
    missing = [l for l in lines if 'REMARK 465' in l]
    for mline in missing:
        splitline = mline.split()
        if len(splitline) > 3:
            resname = splitline[2]
            if resname in valid_residues and splitline[3] == chain:
                resnum = int(splitline[4])
                missing_dict[resnum] = resname
    return missing_dict

def test_consistency(res_col, idx_col):
    """
    This function determins if the idx_col is consistent with being a list of indexes for each residue
    """
    tuple_list = list(zip(idx_col, res_col))
    idx_set = set(idx_col)
    if len(idx_set) < 9:
        return False
    for idx in idx_set:
        if len(set([t[1] for t in tuple_list if t[0]==idx]))>1:
            return False
    return True

def column_identifier(lines, chain):
    '''
    This function determines the index of the column that contains the residue indexes
    as this varies between PDB files
    '''
    atoms = [l for l in lines if 'ATOM ' in l and l.split()[4]==chain]
    num_2_res = {}
    columns_per_line = set([len(atom_line.split()) for atom_line in atoms])
    assert len(columns_per_line) == 1
    columns_per_line = columns_per_line.pop()
    possible_idx_cols = []
    for i in range(columns_per_line):
        col_vals = [atom_line.split()[i] for atom_line in atoms]
        if len(set(col_vals).intersection(set(valid_residues))) > 3:
            residue_column = i
        if all([val.strip('-').isnumeric() for val in atom_line.split()[i] for atom_line in atoms]):
            if all(['.' not in val for val in atom_line.split()[i] for atom_line in atoms]):
                possible_idx_cols.append(i)
    for candidate_col in possible_idx_cols:
        res_col = [atom_line.split()[residue_column] for atom_line in atoms]
        idx_col = [atom_line.split()[candidate_col] for atom_line in atoms]
        if test_consistency(res_col, idx_col):
            return dict(zip(idx_col, res_col))




def get_res_index_from_lines2(lines, chain):
    start_cycle = False
    start_idx = 0
    end_idx = 0
    for i, l in enumerate(lines):
        words = l.split()
        if 'SSSEQI' in l and 'COMPONENT' in l and 'CODE' in l:
            start_idx = i+1
            end_idx = i+1
            start_cycle = True
            first_row = True
        if start_cycle:
            if first_row:
                first_row = False
                continue
            if len(words) > 6 and words[2].isnumeric() and (words[4].isnumeric() or words[4][0]=='-') and words[6].isnumeric() and words[7].isnumeric():
                end_idx +=1
            else:
                break
    relevant_lines = lines[start_idx:end_idx]
    lines_match_chain = [l for l in relevant_lines if l.split()[3]==chain]
    if len(lines_match_chain)>0:
        lines_match_chain[0].split()
    else:
        breakpoint_var = True



def get_res_index_from_lines(lines, chain):
    """
    This function does not work because some pdb files label every atom with a different number
    even if they are part of the same residue
    """
    atoms = [l for l in lines if 'ATOM ' in l and l.split()[4]==chain]
    num_2_res = {}
    for line in atoms:
        l_split = line.split()
        number = l_split[1]
        resname = l_split[3]
        if number in num_2_res:
            assert num_2_res[number] == resname
        else:
            num_2_res[number] = resname
    return num_2_res


def get_chain_letter(domain_id):
    """
    This function should not be required after refactor
    """
    one_chain = df[df.domain.str.contains(domain_id.lower())]
    return one_chain.iloc[0].domain[4]


def count_all_residues(alpha_rep, domain_id, chain, seq_df):
    alpha_count = alpha_rep['single'].shape[0]
    seq_count = len(seq_df[(seq_df.PDBID.str.contains(domain_id.lower()))&(seq_df.CHAIN == chain)].sequence.values[0])
    one_domain = df[df.domain.str.contains(domain_id.lower()+ chain)]
    try:
        assert len(one_domain.domain_length.unique()) == 1
    except AssertionError:
        print(f'multiple domains: {one_domain.domain_length.unique()}')
    domain_str_lst = list(set([r.split('_')[0] for r in one_domain.residue_string.unique()]))
    try:
        assert len(domain_str_lst) == 1
    except AssertionError:
        print(f'multiple domain strings: {domain_str_lst}')
    domain_str = domain_str_lst[0]
    df_count = one_domain.domain_length.unique()[0]
    all_match = seq_count == df_count == alpha_count
    alpha_eq_seq = alpha_count == seq_count
    df_eq_seq = df_count == seq_count
    one_row = {
        'domain_str': domain_str,
        'seq_count': seq_count,
        'df_count': df_count,
        'alpha_count': alpha_count,
        'all_match': all_match,
        'alpha_eq_seq': alpha_eq_seq,
        'df_eq_seq': df_eq_seq,
    }
    return one_row

def add_sequence_to_alpha_rep(alpha_rep, domain_id, chain, seq_df):
    match = seq_df[(seq_df.PDBID == domain_id.lower())&(seq_df.CHAIN == chain)]
    sequence = match.iloc[0].sequence
    alpha_rep['sequence'] = sequence
    if len(sequence) != alpha_rep['single'].shape[0]:
        breakpoint_var = True
        return 'error'
    xs = sequence.count('X')
    return alpha_rep



if __name__=="__main__":
    count_rows = []
    success = 0
    mismatch = 0
    seq_df = pd.read_csv('/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/experiments/PPI_training_dataset_with_sequences.csv')
    dir = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/alpha_pickles/representation_pickles'
    df = pd.read_csv('../datasets/PPI/PPI_training_dataset.csv')
    # atom_groups = get_all_atom_groups(df.iloc[:50,])
    atom_groups = load_atom_groups_from_pickle()
    mismatch_counter = 0
    for i, alpha_domain in enumerate(os.listdir(dir)):
        domain_id = alpha_domain.split('.pickle')[0][:4]
        chain = get_chain_letter(domain_id)
        lines = load_pdb_lines(domain_id)
        missing = get_missing_from_lines(lines, chain)
        num_2_res = get_res_index_from_lines2(lines, chain)
        if i % 10==0:
            print(i)
    #     seq = get_sequence_from_lines(lines, chain)
    #     alpha_rep = load_rep(domain_id + chain)
    #     if alpha_rep is not None:
    #         one_row = count_all_residues(alpha_rep, domain_id, chain, seq_df)
    #         alpha_rep = add_sequence_to_alpha_rep(alpha_rep, domain_id, chain, seq_df)
    #         count_rows.append(one_row)
    #         if isinstance(alpha_rep, str):
    #             mismatch_counter += 1
    #     # if alpha_rep is not None:
    #     #     if len(seq) == alpha_rep.shape[0]:
    #     #         success +=1
    #     #     else:
    #     #         mismatch +=1
    #     # breakpoint_var = True
    #     if i % 50 == 0:
    #         count_res_df = pd.DataFrame(count_rows)
    #         count_res_df.to_csv('count_residues.csv')
    # count_res_df = pd.DataFrame(count_rows)
    # count_res_df.to_csv('count_residues.csv')
    # check_true_length()
    # breakpoint_var = True



