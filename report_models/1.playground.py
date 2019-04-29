"""
Quadratic Regression Patch 1.0

Patch notes:

Rui Nian
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import gc
import argparse

import os

from sklearn.model_selection import train_test_split

import seaborn as sns

import pickle

import warnings

import sys
sys.path.insert(0, '/home/rui/Documents/Willowglen/Suncor_Phase2')

from EWMA import ewma
from Seq_plot import seq_pred
from MinMaxNorm import MinMaxNormalization

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

sns.set()
sns.set_style('white')

gc.enable()

# MacOS & Ubuntu 18.04 path
# path = '/Users/ruinian/Documents/Willowglen/data/Optimization_Data/'
path = '/home/rui/Documents/Willowglen/data/report_datasets/'

"""
Squared data
"""

# data = pd.read_csv(path + 'ls_data.csv')
# test_data = pd.read_csv(path + 'ls_test_data.csv')
#
# header = list(data)
# del header[0]
#
# for name in header:
#     data[name + '_sq'] = np.square(data[name])
#     test_data[name + '_sq'] = np.square(test_data[name])
#
# data.to_csv('ls_data_squared.csv', index=False)
# test_data.to_csv('ls_test_data_squared.csv', index=False)
#
# """
# Square root data
# """
#
# data = pd.read_csv(path + 'ls_data.csv')
# test_data = pd.read_csv(path + 'ls_test_data.csv')
#
# for name in header:
#     data[name + '_sq'] = np.sqrt(data[name])
#     test_data[name + '_sq'] = np.sqrt(test_data[name])
#
# data.to_csv('ls_data_sqrt.csv', index=False)
# test_data.to_csv('ls_test_data_sqrt.csv', index=False)

test_data = pd.read_csv(path + 'ls_set2.csv')
test_data = test_data.iloc[:34500, :]

test_data.to_csv('ls_set2_train.csv', index=False)

