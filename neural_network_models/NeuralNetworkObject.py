"""
Neural Network Object v1.0

Rui Nian

Patch notes:

Current issues:
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf

import gc

import os

from sklearn.model_selection import train_test_split

import pickle

import warnings

import sys

sys.path.insert(0, '/home/rui/Documents/Willowglen/Suncor_Phase2')

from EWMA import ewma
from Seq_plot import seq_pred
from MinMaxNorm import MinMaxNormalization

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# Set seaborn plots
sns.set()
sns.set_style('white')

# Enable garbage collector
gc.enable()


def random_seed(seed):
    """
    Description
       ---
          Sets random seed for NumPy and TensorFlow for replicable results


    Inputs
       ---
          seed: Random seed used for the remainder of the script

    """
    np.random.seed(seed)
    tf.set_random_seed(seed)


def save(obj, path):
    """
    Description
       ---
          Pickle out Python objects (save it as a string representation to load at high speed in the future).  Will
          mainly be used for pickling normalizations.


    Inputs
       ---
           obj: Object to be pickled out.
          path: Location of where to save the pickle.

    """
    pickle_out = open(path, 'wb')
    pickle.dump(obj, pickle_out)
    pickle_out.close()


def load(path):
    """
    Description
       ---
          Load in previously saved objects.  Mainly used for loading in normalization.


    Inputs
       ---
          path: Location of where to load the pickle.  Should be identical to save path.


    Returns
       ---
           obj: Loaded pickle object.

    """
    pickle_in = open(path, 'rb')
    obj = pickle.load(pickle_in)
    return obj


class NeuralNetwork:
    """
    Description
       ---
       Feed-forward neural network with 3 hidden layers built in TensorFlow


    Attributes
       ---
              session: Tensorflow session to be passed into the regression

              train_x: Training features
               test_x: Testing features
              train_y: Corresponding training labels
               test_y: Corresponding test labels

                   lr: Learning rate, alpha \in (0, 1]
       minibatch_size: Mini-batch gradient descent size
           train_size: % of data used for training
               epochs: Number of passes through the whole data set
                lambd: Normalization parameter for L1 or L2 normalization


    Methods
       ---
                train: Runs optimization on the ML model (adaptative momentum gradient descent)
                 test: Tests current model
           loss_check: Checks the current loss of the model
   weights_and_biases: Returns the weights and biases of the model

    """

    def __init__(self):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass


if __name__ == "__main__":

    # Seed NumPy and TensorFlow with the meaning of life again
    random_seed(42)

    # Specify data, model and normalization paths
    Data_path = '/home/rui/Documents/Willowglen/data/Optimization_Data/Opti_withAllChangable.csv'
    Model_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/neural_network_models/checkpoints/ls.ckpt'
    Norm_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/neural_network_models/normalization/ls.pickle'

    Raw_data, Heading_names, Linear_reg, Weights_biases = simulation(Data_path, Model_path, Norm_path, test_size=0.05,
                                                                     shuffle=True, lr=0.003, minibatch_size=2048,
                                                                     train_size=0.9, epochs=30, lambd=0.001,
                                                                     testing=True)
