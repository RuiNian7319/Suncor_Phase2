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
             h1_nodes: Amount of neurons in 1st hidden layer
             h2_nodes: Amount of neurons in 2nd hidden layer
             h3_nodes: Amount of neurons in 3rd hidden layer


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
                 epochs=5, lambd=0.001, h1_nodes=30, h2_nodes=30, h3_nodes=30):

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
        self.input_size = self.nx
        self.h1_nodes = h1_nodes
        self.h2_nodes = h2_nodes
        self.h3_nodes = h3_nodes
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
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

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

        self.sess.run(self.optimizer, feed_dict={self.y: labels, self.X: features, self.training: training})

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


def train_model(data_path, model_path, norm_path, test_size=0.05, shuffle=True, lr=0.003, minibatch_size=2048,
                epochs=30, lambd=0.001, h1_nodes=30, h2_nodes=30, h3_nodes=30,
                testing=True, loading=False):

    """
    Description
       ---
          Trains a normalized (min-max) three-layer feedforward neural network model, given the data from data_path.
          Model will be saved to model_path.  Advanced settings are set above.


    Inputs
       ---
              data_path: Path for the process data.  First column should be labels
             model_path: Path for the model saving.
              norm_path: Path for the normalization object.
              test_size: Size of testing data set
                shuffle: Boolean, shuffle the data for training?  Breaks time correlation of data
                     lr: Learning rate of the model, higher learning rate results in faster, more unstable learning.
         minibatch_size: Size of batches for stochastic / minibatch gradient descent
                 epochs: Number of passes through the whole data
                  lambd: Regularization term
               h1_nodes: Amount of neurons in 1st hidden layer
               h2_nodes: Amount of neurons in 2nd hidden layer
               h3_nodes: Amount of neurons in 3rd hidden layer
                testing: Training or testing?
                loading: True if you want to load an old model for further training


    Returns
       ---
               raw_data: Data used for model building
          heading_names: Headings of the raw data
             linear_reg: Linear regression object

    """

    # Load data
    raw_data = pd.read_csv(data_path)

    # Get heading tags / names, then transform into NumPy array
    heading_names = list(raw_data)
    raw_data = raw_data.values

    print('There are {} feature(s) and {} label(s) with {} examples.'.format(raw_data.shape[1] - 1, 1,
                                                                             raw_data.shape[0]))

    # Train / test split
    train_x, test_x, train_y, test_y = train_test_split(raw_data[:, 1:], raw_data[:, 0], test_size=test_size,
                                                        shuffle=shuffle, random_state=42)

    train_x = train_x.reshape(-1, raw_data.shape[1] - 1)
    test_x = test_x.reshape(-1, raw_data.shape[1] - 1)

    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    # Normalization
    if testing:
        min_max_normalization = load(norm_path)

    else:
        min_max_normalization = MinMaxNormalization(np.concatenate([train_y, train_x], axis=1))

    training_data = min_max_normalization(np.concatenate([train_y, train_x], axis=1))
    testing_data = min_max_normalization(np.concatenate([test_y, test_x], axis=1))

    train_x = training_data[:, 1:].reshape(-1, raw_data.shape[1] - 1)
    test_x = testing_data[:, 1:].reshape(-1, raw_data.shape[1] - 1)

    train_y = training_data[:, 0].reshape(-1, 1)
    test_y = testing_data[:, 0].reshape(-1, 1)

    # Test cases for NaN cases
    assert(not np.isnan(train_x).any())
    assert(not np.isnan(test_x).any())

    assert(not np.isnan(train_y).any())
    assert(not np.isnan(test_y).any())

    with tf.Session() as sess:

        # Build the neural network object
        nn = NeuralNetwork(sess, train_x, train_y, test_x, test_y, lr=lr, minibatch_size=minibatch_size,
                           train_size=1 - test_size, h1_nodes=h1_nodes, h2_nodes=h2_nodes, h3_nodes=h3_nodes,
                           epochs=epochs, lambd=lambd)

        # If testing, just run the model
        if testing:

            # Restore model
            nn.saver.restore(sess, model_path)

            # Predict based on model
            pred = nn.test(features=test_x, training=not testing)

            # Unnormalize
            pred = min_max_normalization.unnormalize_y(pred)

            # Evaluate loss
            rmse, mae = nn.eval_loss(pred, test_y)

            print('Test RMSE: {:2f} | Test MAE: {:2f}'.format(rmse, mae))

        else:

            # If loading old model for continued training
            if loading:
                nn.saver.restore(sess, Model_path)

            else:
                # Initialize global variables
                sess.run(nn.init)

            for epoch in range(epochs):

                for i in range(nn.total_batch_number):

                    # Minibatch gradient descent
                    batch_index = int(i * nn.total_batch_number)

                    minibatch_x = train_x[batch_index:batch_index + nn.minibatch_size, :]
                    minibatch_y = train_y[batch_index:batch_index + nn.minibatch_size, :]

                    # Optimize the machine learning model
                    nn.train(features=minibatch_x, labels=minibatch_y, training=not testing)

                    # Record loss
                    if i % 10 == 0:
                        _ = nn.loss_check(features=train_x, labels=train_y, training=not testing)

                    # Evaluate training and testing losses
                    if i % 150 == 0:

                        # Check for loss
                        current_loss = nn.loss_check(features=test_x, labels=test_y, training=not testing)

                        # Predict training loss
                        pred = nn.test(features=train_x, training=not testing)

                        # Unnormalize
                        pred = min_max_normalization.unnormalize_y(pred)
                        label_y = min_max_normalization.unnormalize_y(train_y)

                        # Evaluate training loss
                        train_rmse, _ = nn.eval_loss(pred, label_y)

                        # Predict test loss
                        pred = nn.test(features=test_x, training=not testing)

                        # Unnormalize
                        pred = min_max_normalization.unnormalize_y(pred)
                        label_y = min_max_normalization.unnormalize_y(test_y)

                        # Evaluate training loss
                        test_rmse, _ = nn.eval_loss(pred, label_y)

                        print('Epoch: {} | Loss: {:2f} | Train RMSE: {:2f} | Test RMSE: {:2f}'.format(epoch,
                                                                                                      current_loss,
                                                                                                      train_rmse,
                                                                                                      test_rmse))

            # Save the model
            nn.saver.save(sess, model_path)
            print("Model saved at: {}".format(model_path))

            # Save normalizer
            save(min_max_normalization, norm_path)
            print("Normalization saved at: {}".format(norm_path))

            # Final test
            test_pred = nn.test(features=test_x, training=not testing)

            # Unnormalize data
            test_pred = min_max_normalization.unnormalize_y(test_pred)
            actual_y = min_max_normalization.unnormalize_y(test_y)

            test_rmse, test_mae = nn.eval_loss(test_pred, actual_y)
            print('Final Test Results:  Test RMSE: {:2f} | Test MAE: {:2f}'.format(test_rmse, test_mae))

    return raw_data, heading_names, nn


if __name__ == "__main__":

    # Seed NumPy and TensorFlow with the meaning of life again
    random_seed(42)

    # Specify data, model and normalization paths
    Data_path = '/home/rui/Documents/Willowglen/data/2019Optimization_Data/' \
                '2019AllData.csv'
    Model_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/neural_network_models/checkpoints/nn2019.ckpt'
    Norm_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/neural_network_models/normalization/nn2019.pickle'

    Raw_data, Heading_names, NN = train_model(Data_path, Model_path, Norm_path, test_size=0.05,
                                              shuffle=True, lr=0.001, minibatch_size=2048,
                                              epochs=200, lambd=0.001, h1_nodes=30, h2_nodes=30,
                                              h3_nodes=30, testing=False, loading=True)
