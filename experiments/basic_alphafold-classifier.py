import pandas as pd
from experiments.add_geometricus import add_residue_col
import xgboost as xgb
import numpy as np
import re
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc



def domain_train_test_split(df):
    train_prop = 0.7
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

def fit_and_evaluate(train, test, target='res_label'):
    train_x = extract_embedding_features(train)
    test_x = extract_embedding_features(test)
    train_y = train[target]
    test_y = test[target]
    model = xgb.XGBClassifier()
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    probas = model.predict_proba(test_x)[:,1]
    print(preds[:10], probas[:10])
    precision, recall, thresholds = precision_recall_curve(test_y, probas)
    pr_auc = auc(recall, precision)
    print(f'accuracy: {accuracy_score(test_y, preds)}')
    print(f'ROC AUC: {roc_auc_score(test_y, probas)} ')
    print(f'PR AUC: {pr_auc} ')
    breakpoint_var = True


def main():
    target = 'annotation_IBIS_PPI_INTERCHAIN'
    train, test = prepare_df(target=target)
    fit_and_evaluate(train, test, target=target)

if __name__=='__main__':
    main()




