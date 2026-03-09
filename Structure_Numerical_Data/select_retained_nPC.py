import numpy as np


def select_retained_nPC(variances, nPC_select_algorithm):
    if nPC_select_algorithm == 'eigengap':
        
        diffs = np.diff(variances)
        
        n_pc1 = np.argmax(np.abs(diffs)) + 1
    
    return [n_pc1]