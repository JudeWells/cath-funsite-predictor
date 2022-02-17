import pandas as pd
from experiments.add_geometricus import add_residue_col
import xgboost as xgb
import numpy as np
import re
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc


features_ff = ['scons','avg_scons','sc5_gs','sc5_scons', 'conserved_hotspot_struc_neighbourhood', 'conserved_surface_hotspot_struc_neighbourhood', 'highly_conserved_struc_neighbourhood', 'highly_conserved_surface_struc_neighbourhood', 'pocket_conserved_struc_neighbourhood', 'pocket_surface_conserved_struc_neighbourhood',
               'avg_charged', 'avg_cx', 'avg_dpx', 'avg_electric_effect', 'avg_flexibility', 'avg_hydropathicity', 'avg_hydrophobicity', 'avg_polarity', 'avg_surface_residues', 'avg_surrounding_hydrophobicity', 'dist_to_hotspot', 'dist_to_surface', 'hotspot_struc_neighbourhood', 'pocket_struc_neighbourhood', 'surface_residues_struc_neighbourhood', 'min_dist_to_cleft123', 'min_dist_to_cleft_1', 'min_dist_to_cleft_2', 'min_dist_to_cleft_3', 'surface_residue_rsa', 'cleft_residue', 'hydrophobic_aa', 'polar_aa',
               'alpha','betweenness','bulkiness','charge','cleft_depth','cleft_num','closeness','degree','foldx_alascan','free_energy_solution','hydration_potential','hydropathicity','hydropathy_index','hydrophobicity','hydrophobicity_psaia','kappa','localised_electrical_effect','max_cx','max_dpx','min_cx','min_dpx','nhBonds_ptr','oBonds_ptr','phi','polarity','psi','resTco','res_bfactor_n','rsa_allatoms','rsa_mainchain','rsa_nonpolar','rsa_polar','rsa_totside','van_der_waals_vol_normalised',
               'dssp_type_B','dssp_type_H','dssp_type_NO_PRED','dssp_type_T',
               'A_pssm_ff','A_wop_ff','C_pssm_ff','C_wop_ff','D_pssm_ff','D_wop_ff','E_pssm_ff','E_wop_ff','F_pssm_ff','F_wop_ff','G_pssm_ff','G_wop_ff','H_pssm_ff','H_wop_ff','I_pssm_ff','I_wop_ff','K_pssm_ff','K_wop_ff','L_pssm_ff','L_wop_ff','M_pssm_ff','M_wop_ff','N_pssm_ff','N_wop_ff','P_pssm_ff','P_wop_ff','Q_pssm_ff','Q_wop_ff','R_pssm_ff','R_wop_ff','S_pssm_ff','S_wop_ff','T_pssm_ff','T_wop_ff','V_pssm_ff','V_wop_ff','W_pssm_ff','W_wop_ff','Y_pssm_ff','Y_wop_ff','gapless_match_to_pseudocounts_ff','info_per_pos_ff',
              ]

def domain_train_test_split(df, split_by_domain=True):
    train_prop = 0.7

    if split_by_domain:
        domains = df.domain.unique()
        np.random.shuffle(domains)
        n_train_domains = int(train_prop*len(domains))
        train_domains = domains[:n_train_domains]
        train = df[df.domain.isin(train_domains)]
        test = df[~df.domain.isin(train_domains)]
    else:
        last_train_idx = int(train_prop * len(df))
        train = df.iloc[:last_train_idx]
        test = df.iloc[last_train_idx:]
    return train, test

def drop_other_columns(df):
    drop_cols = [c for c in df.columns if c not in ['alpha_rep', 'res_label']]

def prepare_df(path_to_df = None, target='res_label'):
    if path_to_df is None:
        path_to_df = '/Users/judewells/Documents/dataScienceProgramming/cath-funsite-predictor/experiments/with_alphafold.csv'
    df = pd.read_csv(path_to_df)
    df = add_residue_col(df)
    print(f'proportion of rows with alpha_rep {df.alpha_rep.notnull().mean()}')
    df.dropna(inplace=True, subset=['alpha_rep', target])
    train, test = domain_train_test_split(df)
    return train, test

def convert_rep_to_numpy(repstr):
    return np.array(eval(re.sub('\[,', '[', re.sub('\s+', ',', repstr))))


def extract_embedding_features(df_alpha):
    return np.stack(df_alpha.alpha_rep.apply(convert_rep_to_numpy))

def fit_and_evaluate(train, test, target='res_label', use_alphafold=True):

    ff_train_x = train[features_ff]
    ff_test_x = test[features_ff]

    if use_alphafold:
        train_x = extract_embedding_features(train)
        test_x = extract_embedding_features(test)
        train_x = pd.concat([pd.DataFrame(train_x), ff_train_x.reset_index()], axis=1)
        test_x = pd.concat([pd.DataFrame(test_x), ff_test_x.reset_index()], axis=1)
    else:
        train_x = ff_train_x
        test_x = ff_test_x
    train_y = train[target]
    test_y = test[target]
    model = xgb.XGBClassifier()
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    train_probas = model.predict_proba(train_x)[:,1]
    train_precision, train_recall, train_thresholds = precision_recall_curve(train_y, train_probas)
    train_pr_auc = auc(train_recall, train_precision)
    probas = model.predict_proba(test_x)[:,1]
    print(preds[:10], probas[:10])
    precision, recall, thresholds = precision_recall_curve(test_y, probas)
    pr_auc = auc(recall, precision)
    print(f'DF shape: {train_x.shape}')
    print(f'TRAINING PR AUC: {train_pr_auc}')
    print(f'accuracy: {accuracy_score(test_y, preds)}')
    print(f'ROC AUC: {roc_auc_score(test_y, probas)} ')
    print(f'PR AUC: {pr_auc} ')
    breakpoint_var = True


def main():
    target = 'annotation_IBIS_PPI_INTERCHAIN'
    train, test = prepare_df(target=target)
    for use_alphafold in [True, False]:
        print(f'---USE ALPHAFOLD: {use_alphafold}---')
        fit_and_evaluate(train, test, target=target, use_alphafold=use_alphafold)

if __name__=='__main__':
    main()




