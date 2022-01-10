import pandas as pd
import numpy as np
import os
dfk = pd.read_csv('training_kmers.csv')
dfr = pd.read_csv('training_radii.csv')
train = pd.read_csv('../datasets/PPI/PPI_training_dataset.csv')
print(dfk.shape, dfr.shape)
print(train.shape)
breakpoint_var = True
