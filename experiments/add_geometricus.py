import os.path

import numpy as np
import pandas as pd
from geometricus import MomentInvariants, SplitType
import prody

def align_residue_index(atom_group, res_num):
    """
    Ways in which res_num might need to be adjusted
    Residue index does not start at 0 -> subtract start index from res_num
    Missing residues -> subtract number of missing residues that lie between min(res_num) and resnum
    """
    resnums = atom_group.getData('resnum')
    if resnums[0] != min(resnums):
        print('first resnum is not lowest - using value at index 0')
    start_index = resnums[0]
    missing_residues = set(range(start_index, max(resnums)+1)) - set(resnums)
    n_missing_before = len([r for r in missing_residues if r < res_num])
    aligned_res_num = res_num - start_index
    aligned_res_num = aligned_res_num - n_missing_before
    return aligned_res_num

def add_residue_col(ppi):
    residue_df = ppi[[c for c in ppi.columns if "residue_aa" in c]]
    int_labels  = np.where(residue_df.values == 1)[1]
    int_2_res = dict( zip(range(max(int_labels)+1), [c.split('_')[-1] for c in residue_df.columns]))
    ppi['res_label'] = int_labels
    ppi['res_label'] = ppi['res_label'].replace(int_2_res)
    return ppi

def refactor_invariants(invariants_kmer, invariants_radius):
    invariants = {}
    sequences = {}
    for i in range(len(invariants_kmer)):
        kmer = invariants_kmer[i]
        radius = invariants_radius[i]
        pdb_ref = invariants_kmer[i].name.lower()
        new_dict = {}
        sequences[pdb_ref] = invariants_kmer[i].sequence
        for j in range(len(kmer.sequence)):
            new_dict[j] = {
                'resname': kmer.sequence[j],
                'kmer': kmer.moments[j],
                'radius': radius.moments[j],

            }
        invariants[pdb_ref] = new_dict
    return invariants, sequences


def generate_invariants(ppi_df, pdb_file_dir='pdbs/'):
    atom_groups = []
    invariants_kmer = []
    invariants_radius = []
    if not os.path.exists(pdb_file_dir):
        os.mkdir(pdb_file_dir)
    current_directory = os.getcwd()
    os.chdir(pdb_file_dir)
    # iterates through all rows of dataframe and obtains geometricus features
    for dom_str in ppi_df.domain.unique():
        pdb_id = dom_str[:-3]
        chain = dom_str[-3].upper()
        atom_grp = prody.parsePDB(pdb_id, chain=chain)
        atom_groups.append(atom_grp)
        invariants_kmer.append(MomentInvariants.from_prody_atomgroup(dom_str, atom_grp, split_type=SplitType.KMER, split_size=16))
        invariants_radius.append(MomentInvariants.from_prody_atomgroup(dom_str, atom_grp, split_type=SplitType.RADIUS, split_size=10))
    os.chdir(current_directory)
    return atom_groups, invariants_kmer, invariants_radius

def make_atomgroup_dict(atom_groups):
    atom_group_names = [ag.__str__().split(' ')[1].lower() for ag in atom_groups]
    atom_group_dict = dict(zip(atom_group_names, atom_groups))
    return atom_group_dict

def match_invariants(ppi_df, atom_groups, invariants, kmer_output_filename, radii_output_filename):
    """Matches the geometricus invariants for each residue in the dataframe"""
    dud_row = np.array([-1,-1,-1,-1])
    kmer_rows = []
    radius_rows = []
    atom_group_dict = make_atomgroup_dict(atom_groups)
    for i, row in ppi_df.iterrows():
        try:
            domain = row.domain.lower()
            res_num = row.domain_residue
            domain_dict = invariants[domain]
            atom_group = atom_group_dict[domain[:5]]
            res_idx = align_residue_index(atom_group, res_num)

            kmer_inv = domain_dict[res_idx]['kmer']
            radius_inv = domain_dict[res_idx]['radius']
            resname = domain_dict[res_idx]['resname']
            if resname != row.res_label:
                print(f'mismatch on residue {res_num} in {domain}, {resname}, {row.res_label}')
                kmer_rows.append(dud_row)
                radius_rows.append(dud_row)
                continue
            else:
                print(f'success on {domain}')
                kmer_rows.append(kmer_inv)
                radius_rows.append(radius_inv)
        except Exception as e:
            kmer_rows.append(dud_row)
            radius_rows.append(dud_row)
            print(f'EXCEPTION {type(e)}: {e}')
            breakpoint_var = True
            continue
        if i % 1000 == 0:
            pd.DataFrame(kmer_rows, columns=['k1', 'k2', 'k3', 'k4']).to_csv(kmer_output_filename, index=False)
            pd.DataFrame(radius_rows, columns=['r1', 'r2', 'r3', 'r4']).to_csv(radii_output_filename, index=False)
    kmers_df = pd.DataFrame(kmer_rows, columns=['k1', 'k2', 'k3', 'k4'])
    radii_df = pd.DataFrame(radius_rows, columns=['r1', 'r2', 'r3', 'r4'])
    kmers_df.to_csv(kmer_output_filename, index=False)
    radii_df.to_csv(radii_output_filename, index=False)
    return kmers_df, radii_df

# if __name__ == '__main__':
#     basedir = '../datasets/PPI/'
#     filenames = ['PPI_validation_dataset', 'PPI_training_dataset.csv']
#     for filename in filenames:
#         kmer_output_filename = 'kmers_' + filename
#         radii_output_filename = 'radii_' + filename
#         df = pd.read_csv(basedir + 'PPI_training_dataset.csv')
#         df = add_residue_col(df)
#         atom_groups, invariants_kmer, invariants_radius = generate_invariants(df)
#         invariants, sequences = refactor_invariants(invariants_kmer, invariants_radius)
#         kmers_df, radii_df = match_invariants(df, atom_groups, invariants, kmer_output_filename, radii_output_filename)
