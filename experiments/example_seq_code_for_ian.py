import pandas as pd

def get_sequence(pdb_id, chain):
    """
    Gets the residue sequence that is associated with a PDB structure
    """
    pass

def map_pdb_residue_id_to_seq_index(pdb_id, chain, seq):
    """
    returns dictionary length=n_res_in_pdb {pdb_res1: seq_index, pdb_res2:...}
    """
    pass

path_to_df = '../datasets/PPI/NEEL-generated_dataset/PPI_training_and_validation.csv'
df = pd.read_csv(path_to_df)

for i, row in df.iterrows():
    pdb_id = row.PDBID
    chain = row.CHAIN
    seq = get_sequence(pdb_id, chain)
    seq_mapping = map_pdb_residue_id_to_seq_index(pdb_id, chain, seq)
