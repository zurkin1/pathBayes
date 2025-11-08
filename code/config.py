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
UDP_WEIGHT=0.5 #Equal balance (recommended default).
CPT_ACTIVATION=0.85 #CPT weight for activation interactions. Conditional Probabilty Table.
CPT_INHIBITION=0.15 #CPT weight for inhibition interactions.
CPT_BASELINE=0.5 #Baseline probability. P(child | parent=0)
DEBUG = False


def parallel_apply(df, func, n_cores=None):
    """Applies a function to DataFrame rows in parallel, preserving order."""
    if n_cores is None:
        n_cores = max(1, mp.cpu_count() - 2)  # leave 2 cores free for OS

    with mp.Pool(n_cores) as pool:
        results = list(
            tqdm(
                pool.imap(func, [row for _, row in df.iterrows()]),
                total=len(df),
                desc="Processing samples",
            )
        )

    return pd.DataFrame(results, index=df.index)

def to_prob_logscale(x, vmax=22.0, eps=1e-6):
    p = np.log1p(x) / np.log1p(vmax)   # log(1+x)/log(1+vmax)
    return np.clip(p, eps, 1 - eps)


def to_prob_power(x, vmax=22.0, alpha=0.5, eps=1e-6):
    p = (np.power(x, alpha)) / (np.power(vmax, alpha))
    return np.clip(p, eps, 1 - eps)