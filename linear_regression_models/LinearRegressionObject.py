"""
Linear Regression Object

Patch notes:

Date of last edit: March 6th
Rui Nian

Current issues:
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

import gc
import argparse

import os

from sklearn.model_selection import train_test_split

import pickle

import warnings
warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

class LinearRegression:

    def __init__(self):
