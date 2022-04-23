'''
Snythetic datasets that are simulated in PrivPC paper, and we use them too.
'''

import numpy as np
import pandas as pd
import os
symbol_map = {'o':1,'>':2,'-':3}


def bn_data(name, feature=None, size=10000):
    data = pd.read_csv(os.getcwd()+"/data/"+name+".csv")
    if name in ['asia', 'cancer', 'earthquake','survey']:
        data = data.drop("Unnamed: 0", axis=1)
    data = data.astype('category')
    data = data.apply(lambda x: x.cat.codes)
    data = np.array(data)
    data = data.astype(int)

    if feature is None:
        feature = data.shape[1]

    return data[:size, :feature]

#%%
# Adjacency matrices for PAGs of the above networks. (with causal insufficiency)

# earthquake
e_amat = np.zeros((5,5))
e_amat[0,2] = 2; e_amat[2,0] = 1
e_amat[1,2] = 2; e_amat[2,1] = 1
e_amat[2,3] = 2; e_amat[3,2] = 3
e_amat[2,4] = 2; e_amat[4,2] = 3

# cancer
c_amat = np.zeros((5,5))
c_amat[0,2] = 2; c_amat[2,0] = 1
c_amat[1,2] = 2; c_amat[2,1] = 1
c_amat[4,2] = 2; c_amat[2,4] = 1
c_amat[2,3] = 2; c_amat[3,2] = 1

# asia
a_amat = np.zeros((8,8))
a_amat[0,1] = 1; a_amat[1,0] = 1
a_amat[2,3] = 1; a_amat[3,2] = 1
a_amat[2,4] = 1; a_amat[4,2] = 1
a_amat[1,5] = 2; a_amat[5,1] = 1
a_amat[3,5] = 2; a_amat[5,3] = 1
a_amat[4,7] = 2; a_amat[7,4] = 1
a_amat[5,6] = 2; a_amat[6,5] = 3
a_amat[5,7] = 2; a_amat[7,5] = 3

# survey
s_amat = np.zeros((6,6))
s_amat[0,2] = 2; s_amat[2,0] = 1
s_amat[1,2] = 2; s_amat[2,1] = 1
s_amat[2,3] = 2; s_amat[3,2] = 3
s_amat[2,4] = 2; s_amat[4,2] = 3
s_amat[3,5] = 2; s_amat[5,3] = 3
s_amat[4,5] = 2; s_amat[5,4] = 3


ground_truth_fci_amat = {
    'asia': a_amat,
    'cancer': c_amat,
    'earthquake': e_amat,
    'survey':s_amat
    }



