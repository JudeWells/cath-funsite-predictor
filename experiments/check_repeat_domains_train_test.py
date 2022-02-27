import pandas as pd
import os

dir = '../datasets/PPI/'
train = pd.read_csv(dir + 'PPI_training_dataset.csv')
test = pd.read_csv(dir + 'PPI_validation_dataset.csv')

train_doms = [d[:4] for d in train.domain.unique()]
test_doms = [d[:4] for d in test.domain.unique()]

overlap = set(train_doms).intersection(set(test_doms))

breakpoint_var = True
print(len(overlap))