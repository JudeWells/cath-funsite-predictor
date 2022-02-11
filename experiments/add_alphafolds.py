import numpy as np
import pandas as pd
import pickle
import os
import prody

def load_rep(dom_id):
    """
    This function takes a domain id and returns the corresponding alphafold
    representation for this domain as a numpy.ndarray
    """
    upper_dom_id = dom_id.upper()
    directory = 'representation_pickles/'
    try:
        with open(directory + upper_dom_id + '.pickle', 'rb') as handle:
            representation = pickle.load(handle)
        return representation
    except:
        pass

if __name__=="__main__":
    df = pd.read_csv('../datasets/PPI/PPI_training_dataset.csv')
