import os
import copy
import pandas as pd

"""
This script adds Neel's annotations to the residues as an additional column
The output is saved with the suffix annotated_
"""

true_path = '../datasets/PPI/NEEL-generated_dataset/PPI_training_and_validation.csv'
true_ann = pd.read_csv(true_path)

train_path = 'c3_with_alphafold.csv'
test_path = 'val_c3_with_alphafold.csv'

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

resnums = true_ann.fillna('()').PDB_RESIDUE_NO.apply(lambda x: eval(x))
interface_lookup = dict(zip(true_ann.PDBID + true_ann.CHAIN, resnums))

discrepancies = 0
matches = 0
for i, df in enumerate([train, test]):
    new_df = copy.copy(df)
    new_df['lookup_str'] = new_df.domain.apply(lambda x: x[:5]) + new_df.domain_residue.astype(str)
    new_df['PPI_interface_true'] = new_df.lookup_str.apply(lambda x: int(x[5:]) in interface_lookup[x[:5]]).replace({True: 1, False: 0})
    if i == 0:
        filepath = 'annotated_' + train_path
        print(new_df.shape)
        print(f'train {(new_df.PPI_interface_true == new_df["annotation_IBIS_PPI_INTERCHAIN"]).mean()}')
    else:
        print(new_df.shape)
        print(f'test {(new_df.PPI_interface_true == new_df["annotation_IBIS_PPI_INTERCHAIN"]).mean()}')
        filepath = 'annotated_' + test_path
    new_df.to_csv(filepath)
    del new_df


breakpoint_var = True
breakpoint_var = True

