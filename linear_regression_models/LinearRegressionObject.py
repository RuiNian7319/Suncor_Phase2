"""
Linear Regression Object

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
import argparse

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
    pickle_out = open(obj, 'wb')
    pickle.dump(pickle_out, path)
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
    pickle_out = open(path, 'rb')
    obj = pickle.load(pickle_out)
    return obj


class LinearRegression:

    """
    Description
       ---
       Linear regression Python object, coded in Tensorflow.
       Basic model structure: y = θ0 + θ1 x1 + 02 x2 + ... + θn xn
            Model structure may be edited


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

    def __init__(self, session, train_x, train_y, test_x, test_y, lr=0.003, minibatch_size=64, train_size=0.9, epochs=5,
                 lambd=0.001):

        """
        Notes: - Input must be in shape=[m, Nx]

        """

        # TensorFlow session
        self.sess = session

        # Machine Learning Data
        self.train_X = train_x
        self.test_X = test_x

        self.train_y = train_y
        self.test_y = test_y

        self.nx = train_X.shape[1]
        self.ny = 1
        self.m = train_X.shape[0]
        self.train_size = train_size

        # Machine Learning Parameters
        self.lr = 0.003
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.lambd = lambd

        self.total_batch_number = int((self.m / self.minibatch_size) * self.train_size)

        # Tensorflow variables
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.nx])
        self.y = tf.paceholder(dtype=tf.float32, shape=[None, self.ny])

        self.W = tf.get_variables('Weights', shape=[self.ny, self.nx],
                                  initializer=tf.contrib.layers.xavier_initializer())

        self.b = tf.get_variables('Biases', shape=[self.ny, 1], initializer=constant_initializer())

        # Model
        self.z = tf.matmul(self.X, self.W) + self.b

        # Mean squared error loss function
        self.regularizer = tf.nn.l2_loss(self.W)
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.z) +
                                   self.lambd * self.regularizer)

        # Optimization, Adaptive Momentum Gradient Descent
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # Initializations & saving
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # Loss history
        self.loss_history = []

    def __str__(self):
        return "Linear Regression using {} features.".format(self.nx)

    def __repr__(self):
        return "LinearRegression()"

    def train(self, features, labels):
        """
        Description
           ---
              Train linear regression.  Runs the optimizer on the loss function


        Inputs
           ---
              features: Features of the machine learning model (X)
                labels: Targets / Labels of the machine learning model (y)

        """

        _ = self.sess.run(self.optimizer, feed_dict={self.y: labels, self.X: features})

    def test(self, features):
        """
        Description
           ---
              Test current model on test data.


        Inputs
           ---
              features: Test features of the machine learning model (test_X)


        Returns
           ---
                 preds: Predictions of the target

        """

        return self.sess.run(self.z, feed_dict={self.X: features})

    def loss_check(self, features, labels):
        """
        Description
           ---
              Checks current loss of the model


        Inputs
           ---
              features: Features of the machine learning model (X)
                labels: Targets / Labels of the machine learning model (y)


        Returns
           ---
              cur_loss: Current loss

        """

        cur_loss = self.sess.run(self.loss, feed_dict={self.y: labels, self.X: features})
        self.loss_history.append(cur_loss)
        return cur_loss

    def weights_and_biases(self):
        """
        Description
           ---
              Evaluates the weights and biases.


        Returns
           ---
              weights/biases: Weights and bias of the model

        """

        return self.sess.run([self.W, self.b])

    @staticmethod
    def eval_loss(pred, actual):
        loss = np.sqrt(np.mean(np.square(np.subtract(pred, actual))))


        return loss


if __name__ == "__main__":

    # Specify data, model and normalization paths
    data_path = '/home/rui/Documents/Willowglen/data/Optimization_Data/Opti_withAllChangable.csv'
    model_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/linear_regression_models/checkpoints/ls.ckpt'
    norm_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/linear_regression_models/normalization/ls.pickle'

    random_seed(42)
