import pandas as pd
from experiments.add_geometricus import add_residue_col
import xgboost as xgb
import numpy as np
import re
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc

"""
This script assumes that there are DFs stored on disk that already have alphafold features
where alphafold features are stored as columns labeled 0-384
"""
features_ff = ['scons','avg_scons','sc5_gs','sc5_scons', 'conserved_hotspot_struc_neighbourhood', 'conserved_surface_hotspot_struc_neighbourhood', 'highly_conserved_struc_neighbourhood', 'highly_conserved_surface_struc_neighbourhood', 'pocket_conserved_struc_neighbourhood', 'pocket_surface_conserved_struc_neighbourhood',
               'avg_charged', 'avg_cx', 'avg_dpx', 'avg_electric_effect', 'avg_flexibility', 'avg_hydropathicity', 'avg_hydrophobicity', 'avg_polarity', 'avg_surface_residues', 'avg_surrounding_hydrophobicity', 'dist_to_hotspot', 'dist_to_surface', 'hotspot_struc_neighbourhood', 'pocket_struc_neighbourhood', 'surface_residues_struc_neighbourhood', 'min_dist_to_cleft123', 'min_dist_to_cleft_1', 'min_dist_to_cleft_2', 'min_dist_to_cleft_3', 'surface_residue_rsa', 'cleft_residue', 'hydrophobic_aa', 'polar_aa',
               'alpha','betweenness','bulkiness','charge','cleft_depth','cleft_num','closeness','degree','foldx_alascan','free_energy_solution','hydration_potential','hydropathicity','hydropathy_index','hydrophobicity','hydrophobicity_psaia','kappa','localised_electrical_effect','max_cx','max_dpx','min_cx','min_dpx','nhBonds_ptr','oBonds_ptr','phi','polarity','psi','resTco','res_bfactor_n','rsa_allatoms','rsa_mainchain','rsa_nonpolar','rsa_polar','rsa_totside','van_der_waals_vol_normalised',
               'dssp_type_B','dssp_type_H','dssp_type_NO_PRED','dssp_type_T',
               'A_pssm_ff','A_wop_ff','C_pssm_ff','C_wop_ff','D_pssm_ff','D_wop_ff','E_pssm_ff','E_wop_ff','F_pssm_ff','F_wop_ff','G_pssm_ff','G_wop_ff','H_pssm_ff','H_wop_ff','I_pssm_ff','I_wop_ff','K_pssm_ff','K_wop_ff','L_pssm_ff','L_wop_ff','M_pssm_ff','M_wop_ff','N_pssm_ff','N_wop_ff','P_pssm_ff','P_wop_ff','Q_pssm_ff','Q_wop_ff','R_pssm_ff','R_wop_ff','S_pssm_ff','S_wop_ff','T_pssm_ff','T_wop_ff','V_pssm_ff','V_wop_ff','W_pssm_ff','W_wop_ff','Y_pssm_ff','Y_wop_ff','gapless_match_to_pseudocounts_ff','info_per_pos_ff',
              ]

def domain_train_test_split(df, split_by_domain=True, train_prop = 0.7):

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

def fit_and_evaluate(train, test, target='res_label', use_alphafold=True, validation=None, use_ff=True):
    """
    Note that this version of the function assumes alphafold features have been
    pre-computed and exist in the dataframe as columns labeled 0-383
    """
    if use_ff:
        ff_train_x = train[features_ff]
        ff_test_x = test[features_ff]
    else:
        ff_train_x = ff_test_x = pd.DataFrame()

    if validation is not None:
        if validation == 'test':
            ff_val_x = ff_test_x
        else:
            ff_val_x = validation[features_ff]

    if use_alphafold:
        train_x = train[[str(i) for i in range(384)]]
        test_x = test[[str(i) for i in range(384)]]
        assert all(ff_train_x.columns == ff_test_x.columns)
        if use_ff: # add the funfam features to the alphafold features
            train_x = pd.concat([pd.DataFrame(train_x), ff_train_x], axis=1)
            test_x = pd.concat([pd.DataFrame(test_x), ff_test_x], axis=1)
            if 'index' in train_x.columns:
                train_x.drop('index', axis=1, inplace=True)
                test_x.drop('index', axis=1, inplace=True)
        if validation is not None:
            if validation == 'test': # use the test set for early stopping
                val_x = test_x
            else:
                val_x = validation[[str(i) for i in range(384)]]
                val_x = pd.concat([pd.DataFrame(val_x), ff_val_x.reset_index()], axis=1)
                val_x.drop('index', axis=1, inplace=True)

    else:
        train_x = ff_train_x
        test_x = ff_test_x
        if validation is not None:
            val_x = ff_val_x
    train_y = train[target]
    test_y = test[target]
    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=7,
        gamma=1,
        reg_apha=1,
        objective='binary:logistic',
        scale_pos_weight=4,
        njobs=-1,
    )
    if len(set(test_y)) > 2:
        loss ='merror'
    else:
        loss = 'logloss'
    if validation is not None:
        if validation == 'test':
            val_y = test_y
        else:
            val_y = validation[target]
        eval_set = [(val_x, val_y)]
        model.fit(train_x, train_y, eval_metric=loss,  eval_set=eval_set, early_stopping_rounds=6)
    else:
        model.fit(train_x, train_y, eval_metric=loss)
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
    return probas

def drop_missing_alphafold_rows(df):
    missing_index = df[df.alpha_rep == 'empty'].index
    return df.drop(missing_index)

def drop_inconsistent_rows(df):
    missing_index = df[df.PPI_interface_true != df.annotation_IBIS_PPI_INTERCHAIN].index
    return df.drop(missing_index)

def main():
    target = 'annotation_IBIS_PPI_INTERCHAIN'
    # target = 'PPI_interface_true'
    train = pd.read_csv('c3_with_alphafold.csv')
    train.drop('alpha_rep', axis=1, inplace=True)
    train.dropna(inplace=True)
    # test = drop_missing_alphafold_rows(pd.read_csv('annotated_processed_validation_with_alphafold.csv'))
    train, test = domain_train_test_split(train, split_by_domain=True, train_prop=0.7)
    # train = drop_inconsistent_rows(train)
    # test = drop_inconsistent_rows(test)
    prediction_columns =[]
    for use_alphafold in [True, False]:
        print(f'---USE ALPHAFOLD: {use_alphafold}---')
        predicted_probs = fit_and_evaluate(train, test, target=target, use_alphafold=use_alphafold, use_ff=True)
        pred_colname = 'af_' + str(use_alphafold) + '_pred'
        prediction_columns.append(pred_colname)
        test[pred_colname] = predicted_probs
    test[['domain', 'domain_residue', 'res_label',  target]+prediction_columns].to_csv('annotated_ppi_alphafold_predictions_validation.csv')


if __name__=='__main__':
    main()




