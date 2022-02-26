import pandas as pd

df_path = '../datasets/PPI/NEEL-generated_dataset/PPI_training_and_validation.csv'
df = pd.read_csv(df_path)
df['PDB_RESIDUE_NO'] = df['PDB_RESIDUE_NO'].fillna('()')


ann = pd.read_csv('training_labels.csv')
print(f'Proportion PPI, Neel {ann.PPI_interface_true.mean()} Sayoni: {ann.annotation_IBIS_PPI_INTERCHAIN.mean()}')

disagree_df = ann[ann.PPI_interface_true != ann.annotation_IBIS_PPI_INTERCHAIN]

disagreements = disagree_df.domain.apply(lambda x: x[:5]).values
prop_dict = {}
for domain in disagreements:
    one_domain = ann[ann.domain.str.contains(domain)]
    disagree_one_domain = one_domain[one_domain.PPI_interface_true != one_domain.annotation_IBIS_PPI_INTERCHAIN]
    proportion_rows_not_in_agreement = len(disagree_one_domain)/len(one_domain)
    prop_dict[domain] = proportion_rows_not_in_agreement

false_negatives = len(ann[(ann.PPI_interface_true==1)&(ann.annotation_IBIS_PPI_INTERCHAIN==0)]) / len(ann)
brakpoint_var = True