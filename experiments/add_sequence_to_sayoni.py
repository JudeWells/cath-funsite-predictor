"""
This file was created based on the Get Sequences.ipynb
The notebook version is the one that was actually run to get the sequences.
"""
import time
import os
import pandas as pd
import Bio
from Bio import Entrez,SeqIO

def add_sequence_to_dataset(input_path, output_path, results):
    dfnew = pd.read_csv(input_path, delimiter='\t')
    for i, row in dfnew.iterrows():
        pdbid = row.PDBID
        chain = row.CHAIN
        combined = (pdbid + chain).upper()
        if combined in results:
            continue
        query=fr'{combined}[All Fields] AND pdb[filter]'
        handle=Entrez.esearch(db="protein", term=query)
        records=Entrez.read(handle)
        id_list=records['IdList']
        #print(id_list)
        handle.close()
        if len(id_list) != 1:
            print(f'search returned {len(id_list)} results')
            continue
        each_id = id_list[0]
        fasta=Entrez.efetch(db="protein", id=each_id, rettype="fasta")
        fasta_record=SeqIO.read(fasta, "fasta")
        sequence = str(fasta_record.seq)
        dfnew.loc[i, 'sequence'] = sequence
        seq_dict = {combined: sequence}
        results.update(seq_dict)
        if i % 50 == 0:
            print(i)
            dfnew.to_csv(output_path, index=False)
        time.sleep(0.34)
    dfnew.to_csv(output_path, index=False)

if __name__=="__main__":
    results = {}
    output_path = 'xxx.csv'
    val_data_path = '../datasets/PPI/NEEL-generated_dataset/PPI_validation_dataset_NS.tsv'
    add_sequence_to_dataset(val_data_path, output_path, results=results)
