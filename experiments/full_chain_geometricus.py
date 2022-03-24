import os
import pickle
import numpy as np
import pandas as pd
from add_geometricus import refactor_invariants, generate_invariants


def concatenate_radius_and_kmer_invariants(invariants):
    return {k: [np.concatenate([v[i]['kmer'], v[i]['radius']]) for i in range(len(v))] for k, v in
                           invariants.items()}


def create_invariants_array(invariants, sequences):
    n_proteins = len(invariants)
    n_dim = 8
    feature_array = np.zeros((n_proteins, max_residues, n_dim))
    combined_invariants = concatenate_radius_and_kmer_invariants(invariants)
    for p_idx, pdbid in enumerate(combined_invariants):
        if p_idx % 100 == 0:
            print(p_idx, 'complete')
        residues_in_truncated_chain = min(max_residues, len(combined_invariants[pdbid]))
        for res_idx in range(residues_in_truncated_chain):
            feature_array[p_idx, res_idx, 0:n_dim] = combined_invariants[pdbid][res_idx]
    return feature_array




if __name__ == '__main__':
    max_residues = 250
    basedir = '../datasets/PPI/'
    filename = 'PPI_validation_dataset.csv'
    invariants_path = basedir + 'all_invariants_' + filename.split('.csv')[0] + '.pickle'
    sequence_path = basedir + 'all_sequences_' + filename.split('.csv')[0] + '.pickle'
    feature_array_output_path = basedir + 'geo_feature_array_' + filename.split('.csv')[0] + '.pickle'
    df = pd.read_csv(basedir + filename)
    atom_groups, invariants_kmer, invariants_radius = generate_invariants(df)
    invariants, sequences = refactor_invariants(invariants_kmer, invariants_radius)
    with open(invariants_path, 'wb') as filehandle:
        pickle.dump(invariants, filehandle)
    with open(sequence_path, 'wb') as filehandle:
        pickle.dump(sequences, filehandle)
    invariants_feature_array = create_invariants_array(invariants, sequences)
    with open(feature_array_output_path, 'wb') as filehandle:
        pickle.dump(invariants_feature_array, filehandle)



    breakpoint_v = True






