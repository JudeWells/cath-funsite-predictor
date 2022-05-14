from copy import copy
import re

import pandas as pd

res_file = 'results2.txt'
with open(res_file, 'r') as filehandle:
    lines = filehandle.readlines()

metrics = ['accuracy', 'ROC AUC', 'PR AUC', 'MCC', 'f1']
experiments = []
for l in lines:
    if l[:6] == '---USE':
        lmod = l.replace("',", '')
        split = [w.strip() for w in  re.split(':|,', lmod)]
        d = {
            'alphafold': split[1],
            'geometricus': split[3],
            'features': split[5],
            'ff': split[7],
            'T5': split[9]
        }
    metric = l.split(':')[0]
    if metric in metrics:
        d[metric] = round(float(l.split(':')[1].strip()), 4)
    if metric == 'DF shape':
        d['nfeatures'] = l.split(':')[1].strip().split(', ')[1][:-1]
    if metric == 'f1':
        experiments.append(copy(d))
        d = {}

df = pd.DataFrame(experiments)
df.to_csv('sayoni_annotation_4000_estimators.csv', index=False)