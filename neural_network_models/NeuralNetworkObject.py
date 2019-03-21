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

    def __str__(self):
        return print('Three-layered feed forward neural network with [} inputs'.format(self.nx))

    def __repr__(self):
        return print('NeuralNetwork()')

    def __init__(self, session, train_x, train_y, test_x, test_y, lr=0.003, minibatch_size=2048, train_size=0.9,
                 epochs=5, lambd=0.001):

        """
        Notes: Input data are in shape [m, Nx]

        """

        # TensorFLow session
        self.sess = session

        # Machine Learning Data
        self.train_X = train_x
        self.test_X = test_x

        self.train_y = train_y
        self.test_y = test_y

        self.nx = train_x.shape[1]
        self.ny = 1
        self.m = train_x.shape[0]
        self.train_size = train_size

        # Machine Learning Parameters
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.lambd = lambd

        self.total_batch_number = int((self.m / self.minibatch_size) * self.train_size)

        # Neural Network Parameters
        self.input_size = nx
        self.h1_nodes = 30
        self.h2_nodes = 30
        self.h3_nodes = 30
        self.output_size = 1

        # TensorFlow placeholders
        with tf.name_scope('Inputs'):
            self.X = tf.placeholder(dtype=tf.float32, shape=[None, self.input_size])
            self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.output_size])

        # Used for batch normalization
        self.training = tf.placeholder(dtype=tf.bool, name='training')

        # TensorFlow model variables
        with tf.name_scope('Model'):
            self.hidden_layer1 = {'W': tf.get_variable('h1_weights', shape=[self.input_size, self.h1_nodes],
                                                       initializer=tf.contrib.layers.variance_scaling_initializer()),
                                  'b': tf.get_variable('h1_bias', shape=[self.h1_nodes],
                                                       initializer=tf.contrib.layers.variance_scaling_initializer())}

            self.hidden_layer2 = {'W': tf.get_variable('h2_weights', shape=[self.h1_nodes, self.h2_nodes],
                                                       initializer=tf.contrib.layers.variance_scaling_initializer()),
                                  'b': tf.get_variable('h2_bias', shape=[self.h2_nodes],
                                                       initializer=tf.contrib.layers.variance_scaling_initializer())}

            self.hidden_layer3 = {'W': tf.get_variable('h3_weights', shape=[self.h2_nodes, self.h3_nodes],
                                                       initializer=tf.contrib.layers.variance_scaling_initializer()),
                                  'b': tf.get_variable('h3_bias', shape=[self.h3_nodes],
                                                       initializer=tf.contrib.layers.variance_scaling_initializer())}

            self.output_layer = {'W': tf.get_variable('output_weights', shape=[self.h3_nodes, self.output_size],
                                                      initializer=tf.contrib.layers.variance_scaling_initializer()),
                                 'b': tf.get_variable('output_bias', shape=[self.output_size],
                                                      initializer=tf.contrib.layers.variance_scaling_initializer())}

        self.l1 = tf.add(tf.matmul(self.X, self.hidden_layer1['W']), self.hidden_layer1['b'])
        self.l1 = tf.nn.relu(self.l1)
        self.l1 = tf.layers.batch_normalization(self.l1, training=self.training)

        self.l2 = tf.add(tf.matmul(self.l1, self.hidden_layer2['W']), self.hidden_layer2['b'])
        self.l2 = tf.nn.relu(self.l2)
        self.l2 = tf.layers.batch_normalization(self.l2, training=self.training)

        self.l3 = tf.add(tf.matmul(self.l2, self.hidden_layer3['W']), self.hidden_layer3['b'])
        self.l3 = tf.nn.relu(self.l3)
        self.l3 = tf.layers.batch_normalization(self.l3, training=self.training)

        self.output = tf.add(tf.matmul(self.l3, self.output_layer['W']), self.output_layer['b'])

        # L2 Regularization
        self.trainable_vars = tf.trainable_variables()
        self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in self.trainable_vars if 'bias' not in v.name]) * self.lambd

        # Loss function
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.output) + self.lossL2)

        # Batch Normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss)

        # Loss history
        self.loss_history = []

        # Global variables initializer
        self.init = tf.global_variables_initializer()

        # TensorFlow saver
        self.saver = tf.train.Saver()

    def train(self, features, labels, training):
        """
        Description
           ---
              Train linear regression.  Runs the optimizer on the loss function


        Inputs
           ---
              features: Features of the machine learning model (X)
                labels: Targets / Labels of the machine learning model (y)
              training: Model training or testing, used for batch normalization

        """
        self.sess.run(optimizer, feed_dict={self.y: labels, self.X: features, self.training: training})

    def test(self, features, training):
        """
        Description
           ---
              Test current model on test data.


        Inputs
           ---
              features: Test features of the machine learning model (test_X)
              training: Model training or testing, used for batch normalization


        Returns
           ---
                 preds: Predictions of the target

        """
        return self.sess.run(self.output, feed_dict={self.X: features, self.training: training})

    def loss_check(self, features, labels, training):
        """
        Description
           ---
              Checks current loss of the model


        Inputs
           ---
              features: Features of the machine learning model (X)
                labels: Targets / Labels of the machine learning model (y)
              training: Model training or testing, used for batch normalization


        Returns
           ---
              cur_loss: Current loss

        """
        cur_loss = self.sess.run(self.loss, feed_dict={self.y: labels, self.X: features, self.training: training})
        self.loss_history.append(cur_loss)

        return cur_loss

    @staticmethod
    def eval_loss(pred, actual):
        """
        Description
           ---
              Evaluates RMSE and MAE loss outside the session


        Inputs
           ---
                  pred: Features of the machine learning model (X)
                actual: Targets / Labels of the machine learning model (y)


        Returns
           ---
                  rmse: Root mean squared error
                   mae: Mean absolute error

        """
        error = np.subtract(pred, actual)
        sq_error = np.square(error)
        mean_sq_error = np.mean(sq_error)
        rmse = np.sqrt(mean_sq_error)

        abs_error = np.abs(error)
        mae = np.mean(abs_error)

        return rmse, mae


def train_model(Data_path, Model_path, Norm_path, test_size=0.05, shuffle=True, lr=0.003, minibatch_size=2048,
                train_size=0.9, epochs=30, lambd=0.001, testing=True):

    """
    Description
       ---



    Inputs
       ---
              pred: Features of the machine learning model (X)
            actual: Targets / Labels of the machine learning model (y)


    Returns
       ---
              rmse: Root mean squared error
               mae: Mean absolute error

    """

    return raw_data, heading_names, linear_reg







if __name__ == "__main__":

    # Seed NumPy and TensorFlow with the meaning of life again
    random_seed(42)

    # Specify data, model and normalization paths
    Data_path = '/home/rui/Documents/Willowglen/data/Optimization_Data/Opti_withAllChangable.csv'
    Model_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/neural_network_models/checkpoints/ls.ckpt'
    Norm_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/neural_network_models/normalization/ls.pickle'

    # Raw_data, Heading_names, Linear_reg = train_model(Data_path, Model_path, Norm_path, test_size=0.05,
    #                                                   shuffle=True, lr=0.003, minibatch_size=2048,
    #                                                   train_size=0.9, epochs=30, lambd=0.001,
    #                                                   testing=True)
