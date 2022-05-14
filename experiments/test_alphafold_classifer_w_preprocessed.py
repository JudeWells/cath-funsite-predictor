import time
import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, f1_score
import matplotlib.pyplot as plt
import shap

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

features_generic = [
               'bulkiness','charge','cleft_depth','cleft_num','hydration_potential','hydropathicity','hydropathy_index','hydrophobicity','localised_electrical_effect','max_cx','max_dpx','min_cx','min_dpx','mutability','polarity','psi','resTco','res_bfactor_n','rsa_allatoms','van_der_waals_vol_normalised','dssp_type_B','dssp_type_H','dssp_type_NO_PRED','dssp_type_T',
               'residue_aa_A','residue_aa_C','residue_aa_D','residue_aa_E','residue_aa_F','residue_aa_G','residue_aa_H','residue_aa_I','residue_aa_K','residue_aa_L','residue_aa_M','residue_aa_N','residue_aa_P','residue_aa_Q','residue_aa_R','residue_aa_S','residue_aa_T','residue_aa_V','residue_aa_W','residue_aa_Y',
               'A_pssm_psiblast','A_wop_psiblast','C_pssm_psiblast','C_wop_psiblast','D_pssm_psiblast','D_wop_psiblast','E_pssm_psiblast','E_wop_psiblast','F_pssm_psiblast','F_wop_psiblast','G_pssm_psiblast','G_wop_psiblast','H_pssm_psiblast','H_wop_psiblast','I_pssm_psiblast','I_wop_psiblast','S_pssm_psiblast','K_pssm_psiblast','K_wop_psiblast','L_pssm_psiblast','L_wop_psiblast','M_pssm_psiblast','M_wop_psiblast','R_wop_psiblast','N_pssm_psiblast','N_wop_psiblast','P_pssm_psiblast','P_wop_psiblast','Q_pssm_psiblast','Q_wop_psiblast','R_pssm_psiblast','S_wop_psiblast','T_pssm_psiblast','T_wop_psiblast','V_pssm_psiblast','V_wop_psiblast','Y_wop_psiblast','Y_pssm_psiblast','W_wop_psiblast','W_pssm_psiblast','info_per_pos_psiblast','gapless_match_to_pseudocounts_psiblast','entwop_score_psiblast',
              ]

features_15 = [
                'scons',
                'rsa_allatoms',
                'rsa_totside',
                'avg_cx',
                'res_bfactor_n',
                'closeness',
                'max_cx',
                'surface_residue_rsa',
                'avg_polarity',
                'highly_conserved_surface_struc_neighbourhood',
                'avg_surface_residues',
                'degree',
                'E_wop_psiblast',
                'K_wop_psiblast',
                'avg_dpx'
                ]

features_generic = [f for f in features_generic if f not in features_ff]

combined_features = features_ff + features_generic

def get_feature_importance(model, test_x, experiment_name=''):
    plt.figure(figsize=(25, 100))
    sorted_idx = model.feature_importances_.argsort()
    plt.barh(test_x.columns[sorted_idx], model.feature_importances_[sorted_idx])
    plt.title('xgb built-in feature importance')
    plt.savefig('xgb_builtin_feature_importance' + experiment_name)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test_x)
    shap.summary_plot(shap_values, test_x, plot_type="bar", max_display=40, plot_size=(20,20), show=False)
    fig = plt.gcf()
    fig.savefig('shap_feature_importance_bar' + experiment_name)
    plt.cla()
    plt.clf()
    shap.summary_plot(shap_values, test_x, max_display=40, plot_size=(20,20), show=False)
    fig = plt.gcf()
    fig.savefig('shap_feature_importance_scatter' + experiment_name)

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

def add_seq_embed(train, test):
    seq_train = pd.read_csv('../datasets/PPI/ppi_sequence_embeddings_train.csv')
    seq_test = pd.read_csv('../datasets/PPI/ppi_sequence_embeddings_validation.csv')
    train = pd.concat([train, seq_train], axis=1)
    test = pd.concat([test, seq_test], axis=1)
    return train, test
    pass

def fit_and_evaluate(train, test, target='res_label', use_alphafold=True, validation=None, use_ff=True,
                     use_geometricus=False, use_generic=True, drop_features=None, use_seq_em=False,
                     add_features=None, log_geometricus=False):
    """
    Note that this version of the function assumes alphafold features have been
    pre-computed and exist in the dataframe as columns labeled 0-383
    """
    features = []
    if use_ff:
        features += features_ff
    if use_generic:
        features += features_generic
    if use_alphafold:
        features += [str(i) for i in range(384)]
    if use_geometricus:
        features += ['k1', 'k2', 'k3', 'k4', 'r1', 'r2', 'r3', 'r4']
        if log_geometricus:
            train, test = apply_log_transform_to_geometricus(train, test)
            if validation is not None:
                print('WARNING log transform has not been applied to the geometricus features in the validation set')
    if drop_features is not None:
        features = [f for f in features if f not in drop_features]
    if add_features is not None:
        features += add_features
    if use_seq_em:
        features += [f'e{i}' for i in range(1024)]
        train, test = add_seq_embed(train, test)
    train = train
    test = test
    train_x = train[features]
    test_x = test[features]
    if validation is not None:
        if validation == 'test':
            val_x = test_x
        else:
            val_x = validation[features]

    train_y = train[target]
    test_y = test[target]

    del train
    del test
    model = xgb.XGBClassifier(
        # tree_method='gpu_hist',
        # gpu_id=0,
        n_estimators=1000,
        learning_rate=0.01,
        # subsample=0.7,
        subsample=0.8,
        colsample_bytree=0.8,
        max_depth=7,
        gamma=1,
        reg_apha=1,
        objective='binary:logistic',
        scale_pos_weight=4,
        njobs=-1,
    )
    if len(set(test_y)) > 2: # if doing multiclass classification
        loss ='merror'
    else:
        loss = 'logloss'
    start = time.time()
    if validation is not None:
        if validation == 'test':
            val_y = test_y
        else:
            val_y = validation[target]
        eval_set = [(val_x, val_y)]
        model.fit(train_x, train_y, eval_metric=loss,  eval_set=eval_set, early_stopping_rounds=6)
    else:
        model.fit(train_x, train_y, eval_metric=loss)
    train_time = time.time() - start
    print(f'Training time {train_time}')
    train_probas = model.predict_proba(train_x)[:,1]
    probas = model.predict_proba(test_x)[:,1]
    preds = [1 if p >= 0.5 else 0 for p in probas]
    scores = get_scores(test_y, probas, preds, train_y, train_probas)

    print(f'DF shape: {train_x.shape}')
    print(f'Test shape {test_x.shape}')
    for k, v in scores.items():
        print(f'{k}: {v}')
    scores['probas'] = probas
    get_feature_importance(model, test_x, experiment_name = f'ALPHAFOLD:{use_alphafold} GEOMET:{use_geometricus} FF:{use_ff} SEQ EM:{use_seq_em} GENERIC{use_generic}')
    features_kept = [c for c in train_x.columns if c not in [str(i) for i in range(384)]]
    scores['features'] = ', '.join(features_kept)
    return scores

def drop_missing_alphafold_rows(df):
    if 'alpha_rep' in df.columns:
        df = df.drop('alpha_rep', axis=1)
    missing_index = df[df[[str(i) for i in range(384)]].max(axis=1) == 0].index
    drop_all_zeros = df.drop(missing_index)
    return drop_all_zeros.dropna()

def drop_inconsistent_rows(df):
    missing_index = df[df.PPI_interface_true != df.annotation_IBIS_PPI_INTERCHAIN].index
    return df.drop(missing_index)

def add_geometricus_features(train, test, geo_train_path, geo_test_path):
    train_geometricus = pd.read_csv(geo_train_path)
    geometricus_features = ['k1', 'k2', 'k3', 'k4', 'r1', 'r2', 'r3', 'r4']
    train = train.join(train_geometricus.set_index('residue_string', drop=True)[geometricus_features],
                       on='residue_string', how='left')
    del train_geometricus
    test_geometricus = pd.read_csv(geo_test_path)
    test = test.drop('residue_string', axis=1).join(test_geometricus[geometricus_features], how='left')
    del test_geometricus
    return train, test

def apply_log_transform_to_geometricus(train, test):
    geometricus_features = ['k1', 'k2', 'k3', 'k4', 'r1', 'r2', 'r3', 'r4']
    new_train = train.copy()
    new_test = test.copy()
    for df in [new_train, new_test]:
        for col in geometricus_features:
            df[col] = np.log(df[col], where=df[col]>0, out=np.ones(len(df))*-1)
    return train, test

def get_scores(test_y, probas, preds, train_y, train_probas):
    precision, recall, thresholds = precision_recall_curve(test_y, probas)
    train_precision, train_recall, train_thresholds = precision_recall_curve(train_y, train_probas)

    return {
        'train_pr_auc': auc(train_recall, train_precision),
        'accuracy': accuracy_score(test_y, preds),
        'ROC AUC': roc_auc_score(test_y, probas),
        'PR AUC': auc(recall, precision),
        'MCC': matthews_corrcoef(test_y, preds),
        'f1': f1_score(test_y, preds)
    }


def main():
    # target = 'annotation_IBIS_PPI_INTERCHAIN'
    target = 'PPI_interface_true'
    train = pd.read_csv('annotated_processed_training_with_alphafold.csv')
    test = pd.read_csv('annotated_processed_validation_with_alphafold.csv')
    # geometricus_train_path = '../datasets/PPI/geometricus_PPI_training_dataset.csv'
    # geometricus_test_path = '../datasets/PPI/geometricus_PPI_validation_dataset.csv'
    # train, test = add_geometricus_features(train, test, geometricus_train_path, geometricus_test_path)
    # train = drop_missing_alphafold_rows(train)
    # test = drop_missing_alphafold_rows(test)

    # train = drop_inconsistent_rows(train)
    # test = drop_inconsistent_rows(test)
    prediction_columns =[]
    for use_geometricus in [False]:
        for use_alphafold in [True]:
            for use_ff in [True]:
                for use_seq_em in [False]:
                    for use_generic in [True]:
                        if use_ff == use_geometricus == use_alphafold == use_seq_em == False:
                            continue
                        for log_geometricus in [False]:
                            if log_geometricus and not use_geometricus:
                                continue
                            print(f'---USE ALPHAFOLD: {use_alphafold}, USE GEOMET: {use_geometricus}, \
                            USE FF: {use_ff}, USE SEQ EM: {use_seq_em}, USE GENERIC {use_generic}, LOG GEOMETRICUS: {log_geometricus}---')
                            eval_dict = fit_and_evaluate(train, test, target=target, use_alphafold=use_alphafold, use_ff=use_ff,
                                                         use_geometricus=use_geometricus, use_generic=use_generic, use_seq_em=use_seq_em,
                                                         log_geometricus=log_geometricus)
                            predicted_probs = eval_dict['probas']
                            pred_colname = 'af_' + str(use_alphafold) + '_pred'
                            prediction_columns.append(pred_colname)
                            test[pred_colname] = predicted_probs
    # test[['domain', 'domain_residue', 'res_label',  target]+prediction_columns].to_csv('classifier_results_on_validation.csv', index=False)


if __name__=='__main__':
    main()




