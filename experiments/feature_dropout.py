from test_alphafold_classifer_w_preprocessed import combined_features, fit_and_evaluate, add_geometricus_features, drop_missing_alphafold_rows, get_scores
import pandas as pd


if __name__ == '__main__':
    target = 'PPI_interface_true'
    train = pd.read_csv('annotated_processed_training_with_alphafold_v2.csv')
    test = pd.read_csv('annotated_processed_validation_with_alphafold.csv')
    geometricus_train_path = '../datasets/PPI/geometricus_PPI_training_dataset.csv'
    geometricus_test_path = '../datasets/PPI/geometricus_PPI_validation_dataset.csv'
    train, test = add_geometricus_features(train, test, geometricus_train_path, geometricus_test_path)
    train = drop_missing_alphafold_rows(train)
    test = drop_missing_alphafold_rows(test)
    print(train.shape, test.shape)
    results = pd.DataFrame(index=combined_features, columns=['train_pr_auc','accuracy','ROC AUC', 'PR AUC',])
    for drop_feature in combined_features:
        if 'residue_aa' in drop_feature:
            continue
        eval_dict = fit_and_evaluate(train, test, target=target, use_alphafold=True, use_ff=True,
                                       use_geometricus=False, drop_features=[drop_feature])
        del eval_dict['probas']
        results.loc[drop_feature, list(eval_dict.keys())] = list(eval_dict.values())
        results.to_csv('dropped_feature_analysis.csv')

