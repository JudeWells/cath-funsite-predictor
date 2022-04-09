import os
import numpy as np
import pickle
import pandas as pd
from experiments.parse_chain_gcf import get_gcf_sequences_and_mapping
from experiments.add_geometricus import add_residue_col

embed_dir = 'PPI_sequence_embeddings/'
feature_df = pd.read_csv('../datasets/PPI/PPI_validation_dataset.csv')
feature_df['pdbid'] = feature_df.domain.apply(lambda x: x[:5])
feature_df = add_residue_col(feature_df)
filler_features = np.zeros([len(feature_df), 1024], dtype='float32')

embed_features = [f'e{i}' for i in range(1024)]
embedding_df = pd.DataFrame(filler_features, index=feature_df.index, columns=embed_features)

match_counter = 0
mis_counter = 0
for embed_path in os.listdir(embed_dir):
    pdbid = embed_path.split('.pickle')[0]
    if '.pickle' not in embed_path:
        continue
    with open(embed_dir + embed_path, 'rb') as filehandle:
        embed_dict = pickle.load(filehandle)
    fasta, pdb, mapping, pdb_num_to_letter_dict = get_gcf_sequences_and_mapping(pdbid, pdb_to_letter=True)
    # if embed_dict['sequence'] == fasta:
    if True:
        match_counter += 1
        one_chain = feature_df[feature_df.pdbid == pdbid]
        for i, row in one_chain.iterrows():
            res_letter = row.res_label
            res_id = str(row.domain_residue)
            position_index = mapping[res_id] -1
            if not (embed_dict['sequence'][position_index] == fasta[position_index] == res_letter):
                print(f'non matching residue {pdbid}')
                continue
            embedding_df.loc[i, embed_features] = embed_dict['sequence_embedding'][position_index]
    else:
        mis_counter +=1
embedding_df.to_csv('../datasets/PPI/ppi_sequence_embeddings_validation.csv')
bp=True