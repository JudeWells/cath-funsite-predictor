from test_alphafold_classifer_w_preprocessed import combined_features, fit_and_evaluate, add_geometricus_features, drop_missing_alphafold_rows, get_scores
import pandas as pd


if __name__ == '__main__':
    output_filepath = 'round2_added_feature_analysis.csv'
    target = 'PPI_interface_true'
    train = pd.read_csv('annotated_processed_training_with_alphafold.csv')
    test = pd.read_csv('annotated_processed_validation_with_alphafold.csv')
    geometricus_train_path = '../datasets/PPI/geometricus_PPI_training_dataset.csv'
    geometricus_test_path = '../datasets/PPI/geometricus_PPI_validation_dataset.csv'
    train, test = add_geometricus_features(train, test, geometricus_train_path, geometricus_test_path)
    train = drop_missing_alphafold_rows(train)
    test = drop_missing_alphafold_rows(test)
    print(train.shape, test.shape)
    try:
        results = pd.read_csv(output_filepath).set_index('Unnamed: 0')
    except FileNotFoundError:
        results = pd.DataFrame(index=combined_features, columns=['train_pr_auc','accuracy','ROC AUC', 'PR AUC',])
    # train_reduced = train.drop(combined_features, axis=1)
    # test_reduced = test.drop(combined_features, axis=1)
    alphafold_features = [str(i) for i in range(384)]
    geometricus_features = ['k1', 'k2', 'k3', 'k4', 'r1', 'r2', 'r3', 'r4']
    keep_features = ['rsa_allatoms', 'rsa_totside', 'avg_cx', 'res_bfactor_n', 'closeness', 'max_cx', 'surface_residue_rsa']
    # best features = ['rsa_allatoms', 'rsa_totside', 'avg_cx', 'res_bfactor_n', 'closeness', 'max_cx', 'surface_residue_rsa', 'degree']
    for add_feature in combined_features:
        if add_feature in keep_features:
            continue
        if 'residue_aa' in add_feature:
            continue
        if results.notnull().loc[add_feature, 'ROC AUC']:
            continue
        # train_reduced[add_feature] = train[add_feature]
        # test_reduced[add_feature] = test[add_feature]
        eval_dict = fit_and_evaluate(train, test, target=target, use_alphafold=True, use_ff=False,
                                       use_geometricus=True, use_generic=False, drop_features=None, add_features=[add_feature]+keep_features)
        del eval_dict['probas']
        results.loc[add_feature, list(eval_dict.keys())] = list(eval_dict.values())
        results.to_csv(output_filepath)

