import os
import re
import numpy as np
import pandas as pd

"""This script takes the version of the dataframe with alphafold 
representations saved as a string column and saves it as a dataframe 
where each dimension of the representation is a separate column
"""

def convert_alpha_str_to_columns(df):
    pass

def convert_rep_to_numpy(repstr):
    if repstr == 'empty':
        return np.zeros(384)
    return np.array(eval(re.sub('\[,', '[', re.sub('\s+', ',', repstr))))


def extract_embedding_features(df_alpha):
    df_alpha['alpha_rep'] = df.alpha_rep.fillna('empty')
    return np.stack(df_alpha.alpha_rep.apply(convert_rep_to_numpy))

def check_if_alphafold_exists(domain_str):
    dir1 = '../alpha_pickles/representation_pickles/'
    dir2 = '../alpha_pickles/representation_pickles_model1_validation/'
    pickles = os.listdir(dir1) + os.listdir(dir2)
    search_str = domain_str[:5].upper() + '.pickle'
    if search_str in pickles:
        return 1
    else:
        return 0

train_path = 'training_with_alphafold.csv'
test_path = 'validation_with_alphafold.csv'

def create_missing_report(df):
    missing = df[df.alpha_rep == 'empty']
    missing_domains = missing.domain.unique()
    rows = []
    for d in missing_domains:
        alpha_exists = check_if_alphafold_exists(d)
        new_row = {
            'domain': d,
            'alpha_exists': alpha_exists
        }
        rows.append(new_row)
    report = pd.DataFrame(rows)
    report.to_csv('missing_alphas_report.csv', index=False)



for df_path in [train_path, test_path]:
    df = pd.read_csv(df_path)
    alpha_features = extract_embedding_features(df)
    drop_cols = [c for c in df.columns if c in [ 'index']]
    combined_df = pd.concat([df, pd.DataFrame(alpha_features)], axis=1)
    combined_df = combined_df.drop(drop_cols, axis=1)
    assert len(combined_df.columns) == alpha_features.shape[1] + len(df.columns)
    new_filename = 'processed_' + df_path
    print(f"{df_path} {df.shape}")
    combined_df.to_csv(new_filename)
    create_missing_report(df)
