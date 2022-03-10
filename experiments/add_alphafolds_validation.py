import time
import re
import gzip
import numpy as np
import pandas as pd
import pickle
import os
import prody
from experiments.add_geometricus import add_residue_col
from Bio import pairwise2

"""
This module loads the pre-computed alphafold representations and 
aligns them with the existing dataset and adds the alphafold features
This version of the code uses the index of the PDB file to identify the relevant columns (not splitting on white space)
It also includes rigorous checking to ensure that the indexing does not result in any contrdiction in relation to the amino acid
It also uses the sequence alignment tool where necessary to get a good alignment 
(only uses alignement where the PDB sequence is not a subset of the alphafold sequence 
"""

valid_residues = ['LEU', 'GLU', 'ARG', 'VAL', 'LYS', 'ILE', 'ASP', 'PHE', 'ALA', 'TYR', 'THR', 'SER', 'GLN', 'PRO',
                  'ASN', 'GLY', 'HIS', 'MET', 'TRP', 'CYS']

valid_res_letters = 'GAVCPLIMWFKRHSTYNQDE'

name2letter = dict(zip(valid_residues, 'LERVKIDFAYTSQPNGHMWC'))
letter2name = {v:k for k,v in name2letter.items()}

def load_atom_groups_from_pickle():
    dir = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/experiments/pdbs/'
    with open(dir + 'atom_groups.pickle', 'rb') as handle:
        atom_groups = pickle.load(handle)
    return atom_groups

def download_pdb(domain, chain, output_dir='pdbs'):
    current_dir = os.getcwd()
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    os.chdir(output_dir)
    pdb_id = domain[:4]
    prody.parsePDB(pdb_id, chain=chain)
    os.chdir(current_dir)
    time.sleep(0.33)


def load_rep(dom_id, directory):
    """
    This function takes a domain id and returns the corresponding alphafold
    representation for this domain as a numpy.ndarray
    """
    upper_dom_id = dom_id.upper()[:5]
    # directory = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/alpha_pickles/representation_pickles_model1_validation/'
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


def find_expected_true_length(atom_group):
    valid_residues = ['LEU', 'GLU', 'ARG', 'VAL', 'LYS', 'ILE', 'ASP', 'PHE', 'ALA', 'TYR', 'THR', 'SER', 'GLN', 'PRO',
                      'ASN', 'GLY', 'HIS', 'MET', 'TRP', 'CYS']
    num_2_res = dict(zip(atom_group.getData('resnum'), atom_group.getResnames()))
    refined_num_2_res = {k:v for k,v in num_2_res.items() if v in valid_residues}
    expected_residues = 1 + max(refined_num_2_res.keys()) - min(refined_num_2_res.keys())
    return expected_residues


def gzip_open(filename, *args, **kwargs):
    if args and "t" in args[0]:
        args = (args[0].replace("t", ""), ) + args[1:]
    if isinstance(filename, str):
        return gzip.open(filename, *args, **kwargs)
    else:
        return gzip.GzipFile(filename, *args, **kwargs)


def load_pdb_lines(pdb_code):
    chain = pdb_code[4].upper()
    pdb_code = pdb_code[:4].lower()
    dir = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/experiments/pdbs/'
    filepath = dir + pdb_code.lower() + '.pdb.gz'
    if not os.path.exists(filepath):
        download_pdb(pdb_code, chain)
    with gzip_open(filepath) as handle:
        lines = handle.readlines()
    str_lines = [str(l) for l in lines]
    return str_lines


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
                resnum = splitline[4]
                missing_dict[resnum] = resname
    return missing_dict

def get_missing_from_lines2(lines, chain):
    """
    DO NOT USE
    This function splits on row index but has been shown to be inconsistent:
    some sequence ids start on row 25 and some start on row 26
    and we do not know what happens in files where the sequence ids are 4 characters long
    """
    missing_dict = {}
    missing = [l for l in lines if 'REMARK 465' in l and len(l)>25 and l[26].isnumeric and l[21]==chain and l[17:20] in valid_residues]
    for mline in missing:
        resnum = int(mline[25:].split()[0])
        resname = mline[17:20]
        missing_dict[resnum] = resname
    return missing_dict


def test_consistency(res_col, idx_col):
    """
    This function determins if the idx_col is consistent with being a list of indexes for each residue
    """
    tuple_list = list(zip(idx_col, res_col))
    idx_set = set(idx_col)
    if len(idx_set) > len(idx_col) * 0.5: # the true idx set should be significantly shorter than the entire index col because many repeated idx (res) across atoms
        return False
    if len(idx_set) < 9:
        return False
    for idx in idx_set:
        if len(set([t[1] for t in tuple_list if t[0]==idx]))>1:
            return False
    return True


def combine_missing_atom(missing, num2res):
    if missing is None:
        return num2res
    try:
        num2res_int = {int(k): v for k,v in num2res.items()}
        if missing is None:
            return num2res
        missing_int = {int(k): v for k, v in missing.items()}
        num2res_int.update(missing_int)
        return num2res_int
    except:
        num2res.update(missing)
        return num2res

def atom_identifier(lines, chain):
    lines = [l.strip('b\'') for l in lines]
    lines = [l for l in lines if l[0:4] == 'ATOM']
    lines = [l for l in lines if l[21]==chain]
    res_seq = [l[22:26].strip() for l in lines]
    res_name = [l[17:20].strip() for l in lines]
    if test_consistency(res_name, res_seq):
        return dict(zip(res_seq, res_name))

def make_seqID2res_dict(pdbid, chain):
    """
    aggregator of other functions to create a dictionary of resides from pdb file
    the residue dictionary is a combination of the ATOM lines and also missing residues
    """
    lines = load_pdb_lines(pdbid)
    missing = get_missing_from_lines(lines, chain)
    num2res = atom_identifier(lines, chain)
    combined = combine_missing_atom(missing, num2res)
    return combined


def add_alphafold_rep_alignment(df, mapping, domain, match_status, dir):
    '''
    This version uses a sequence alignment which is encoded in the mapping dictionary
    '''
    one_domain = df[df.domain == domain]
    alpha_rep = load_rep(domain, dir)
    if alpha_rep is None:
        print(f'missing representation {domain}')
        return
    if alpha_rep['single'].shape[0] != len(match_status['alpha_seq']):
        print(f'alpha rep does not match sequence {domain}')
        return
    dimensions = alpha_rep['single'].shape[1]
    for i, row in one_domain.iterrows():
        pdb_num = row.domain_residue
        if isinstance(pdb_num, int) and isinstance(min(mapping.keys()), str):
            pdb_num = str(pdb_num)
        alpha_position = mapping[pdb_num]
        assert row.res_label == match_status['alpha_seq'][alpha_position]
        # df.loc[i, 'alpha_rep'] = str(alpha_rep['single'][alpha_position])
        df.loc[i, [str(n) for n in range(dimensions)]] = alpha_rep['single'][alpha_position] #todo this can be be sped up
    return df


def add_alphafold_rep(df, match_status, seqID2res, domain, dir):

    one_domain = df[df.domain == domain]
    alpha_rep = load_rep(domain, dir)
    if alpha_rep is None:
        print(f'missing representation {domain}')
        return
    if alpha_rep['single'].shape[0] != len(match_status['alpha_seq']):
        print(f'alpha rep does not match sequence {domain}')
        return
    dimensions = len(alpha_rep['single'][0])
    for i, row in one_domain.iterrows():
        pdb_num = row.domain_residue

        if isinstance(pdb_num, int) and isinstance(min(seqID2res.keys()), str):
            pdb_num = str(pdb_num)

        if pdb_num not in seqID2res:
            seqID2res = {re.sub('[A-z]', '', k):v for k,v in seqID2res.items()}
        res_idx_list = np.array(sorted(seqID2res.keys(), key=lambda x: int(x)))
        pdb_resname = name2letter[seqID2res[pdb_num]]
        pdb_position = np.where(res_idx_list == pdb_num)[0][0]
        adjusted_pdb_position = pdb_position + match_status['start_index']
        if not pdb_resname == match_status['pdb_seq'][pdb_position] == match_status['alpha_seq'][adjusted_pdb_position] == row.res_label:
            print('failed to match amino acid type')
            return
        df.loc[i, [str(n) for n in range(dimensions)]] = alpha_rep['single'][adjusted_pdb_position] # todo this can be sped up
    return df

def find_completed_domains(path2df=None, adf=None):
    """this function reads the csv file and returns a list of all the domains
    where the alphafold rep has already been written into the file
    """
    if adf is None:
        adf = pd.read_csv(path2df)
    if '383' in adf.columns:
        completed_domains = adf[adf['383'].notnull()].domain.unique()
        return completed_domains
    else:
        return set()

def process_chain_in_index(seqID2res, chain):
    return {k:v for k,v in seqID2res.items() if chain in k.upper()}

def create_mapping_dict(seqID2res, match_status):
    mapping = {}
    pdb_keys = sorted(int(k) for k in seqID2res.keys())
    alignment = match_status['alignment']
    pdb_counter = 0
    alpha_counter = 0
    for i, align_pdb in enumerate(alignment['seqB']):
        align_alpha = alignment['seqA'][i]
        if align_pdb != '-':
            if align_pdb == align_alpha:
                mapping[pdb_keys[pdb_counter]] = alpha_counter
            pdb_counter +=1
        if align_alpha != '-':
            alpha_counter += 1
    if all([letter2name[alignment['seqA'][mapping[k]]]==v for k,v in seqID2res.items()]):
        return mapping
    else:
        return None


def iterate_and_add(df, seq_df, dir):
    """
    Strategy for matching
    """
    has_alphafold_already = find_completed_domains(adf=df)
    results_list = []
    for i, domain in enumerate(df.domain.unique()):
        if domain in has_alphafold_already:
            continue
        try:
            chain = domain[4]
            seq_alpha = seq_df[(seq_df.PDBID == domain[:4].lower()) & (seq_df.CHAIN == chain)].sequence.min()
            seqID2res = make_seqID2res_dict(domain, chain)



            if any(chain in str(k) for k in seqID2res.keys()):
                seqID2res = process_chain_in_index(seqID2res, chain)
            match_status = get_start_index_and_check_match(seq_alpha, seqID2res)
            match_status['domain'] = domain
            if not match_status['perfect_match']:
                mapping_dict = create_mapping_dict(seqID2res, match_status)
                add_alphafold_rep_alignment(df, mapping_dict, domain, match_status, dir)

            elif match_status['perfect_match']:
                add_alphafold_rep(df, match_status, seqID2res, domain, dir)

        except Exception as E:
            print(f'Error {domain}')
            match_status = {
                'domain': domain,
                'exception': E,
            }
        results_list.append(match_status)
        if i % 20 == 0:
            print(f'{i} domains complete')
        if i % 200 == 0:
            df.to_csv(path_df_w_features, index=False)

    df.to_csv(path_df_w_features, index=False)
    breakpoint_var = True
    results = pd.DataFrame(results_list)
    # results.to_csv('alignment_success_on_pdb_dict.csv', index=False)

def get_alignment(alpha_seq, pdb_seq):
    result = pairwise2.align.globalxx(alpha_seq, pdb_seq)[0]._asdict()
    del_count = result['seqB'].count('-')
    if del_count > 0.2 * len(pdb_seq):
        return None
    return result

def get_start_index_and_check_match(alpha_seq, seqID2res, domain=None, chain=None):
    """This function makes an adjustment at the beginning and the end of the sequence
    counts discontinuities in the middle"""
    pdb_seq = ''.join([name2letter[v] for k,v in sorted(seqID2res.items())])
    idx = 0
    for i in range(20, 1, -1):
        pdb_string  = pdb_seq[:i]
        matches = [(m.start(0), m.end(0)) for m in re.finditer(pdb_string, alpha_seq)]
        if len(matches) > 1:
            breakpoint_var = True
        elif len(matches) == 1:
            break
    start_index = matches[0][0]
    alignment = get_alignment(alpha_seq, pdb_seq)
    if pdb_seq in alpha_seq:
        perfect_match = 1
    else:
        perfect_match = 0
    one_row = {
        'perfect_match': perfect_match,
        'start_index': start_index,
        'alpha_seq': alpha_seq,
        'pdb_seq': pdb_seq,
        'alignment':alignment
    }
    return one_row

def calculate_proportion_of_domains_downloaded(alpha_dir, df):
    downloaded_domains  = set([f.split('.pickle')[0] for f in os.listdir(alpha_dir)])
    dataset_domains = set([domain.upper()[:-2] for domain in df.domain.unique()])
    proportion_downloaded  = len(downloaded_domains.intersection(dataset_domains)) / len(dataset_domains)
    print(proportion_downloaded)


if __name__=="__main__":
    path_df_w_features = 'val_c3_with_alphafold_v2.csv' # this is the path where that dataset with all features will be saved
    seq_df = pd.read_csv('/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/experiments/PPI_validation_w_sequences.csv')
    dir = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/alpha_pickles/c3_combined/'
    df = pd.read_csv('../datasets/PPI/PPI_validation_dataset.csv') # use this if running for the first time
    # df = pd.read_csv(path_df_w_features)
    if 'alpha_rep' not in df.columns:
        df['alpha_rep'] = None
    df = add_residue_col(df)
    iterate_and_add(df, seq_df, dir)


