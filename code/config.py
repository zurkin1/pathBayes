import sys
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import pandas as pd


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

infinitesimal = np.finfo(float).eps
data_path = './data/'
TEST = ''
UDP_WEIGHT=0.5 #Equal balance (recommended default).
CPT_ACTIVATION=0.85 #CPT weight for activation interactions. Conditional Probabilty Table.
CPT_INHIBITION=0.15 #CPT weight for inhibition interactions.
CPT_BASELINE=0.1 #Baseline probability.


def parallel_apply(df, func, n_cores=None):
    """Applies a function to DataFrame rows in parallel, preserving order."""
    if n_cores is None:
        n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    
    results = []
    for result in tqdm(pool.imap(func, [row for _, row in df.iterrows()]), total=len(df), desc="Processing samples"):
        results.append(result)
    
    pool.close()
    pool.join()

    return pd.DataFrame(results, index=df.index)