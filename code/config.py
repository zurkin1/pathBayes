import sys
import os
import numpy as np


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