# libraries
import pickle
import os
import numpy as np
from bio_embeddings.embed import ProtTransT5BFDEmbedder
import pandas as pd
from experiments.parse_chain_gcf import get_gcf_sequences_and_mapping

def pickler(embedding_dict, output_dir, chain_id):
	filename = output_dir + chain_id + '.pickle'
	with open(filename, 'wb') as filehandle:
		pickle.dump(embedding_dict, filehandle)


def unpickler(dir, chain_id):
	if '.pickle' not in chain_id:
		chain_id += '.pickle'
	with open(dir + chain_id, 'rb') as filehandle:
		return pickle.load(filehandle)

if __name__=="__main__":
	embedder = ProtTransT5BFDEmbedder()

	df = pd.read_csv('../datasets/PPI/NEEL-generated_dataset/PPI_validation_dataset_NS.tsv', delimiter='\t')
	output_dir = 'PPI_sequence_embeddings/'
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)


	for i, row in df.iterrows():
		# seq = row.sequence

		chain_id = row.PDBID + row.CHAIN
		fasta, pdb, mapping = get_gcf_sequences_and_mapping(chain_id)
		seq = fasta
		embed_chain = np.asarray(embedder.embed(seq))
		embed_dict = {
			'sequence': seq,
			'sequence_embedding': embed_chain
		}
		pickler(embed_dict, output_dir, chain_id)
