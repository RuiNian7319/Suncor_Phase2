"""
Linear Regression Object v1.0

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

         num_of_const: Numer of weights constrained

    Methods
       ---
                train: Runs optimization on the ML model (adaptative momentum gradient descent)
                 test: Tests current model
           loss_check: Checks the current loss of the model
   weights_and_biases: Returns the weights and biases of the model

    """

    def __init__(self, session, train_x, train_y, test_x, test_y, lr=0.003, minibatch_size=64, train_size=0.9, epochs=5,
                 lambd=0.001, num_of_const=10):

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

        self.nx_Cons = num_of_const
        self.nx_noCons = train_x.shape[1] - self.nx_Cons
        self.ny = 1
        self.m = train_x.shape[0]
        self.train_size = train_size

        # Machine Learning Parameters
        self.lr = lr
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.lambd = lambd

        self.total_batch_number = int((self.m / self.minibatch_size) * self.train_size)

        # Tensorflow variables
        self.X1 = tf.placeholder(dtype=tf.float32, shape=[None, self.nx_Cons])
        self.X2 = tf.placeholder(dtype=tf.float32, shape=[None, self.nx_noCons])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.ny])

        self.W1 = tf.get_variable('Weights_Const', shape=[self.nx_Cons, self.ny],
                                  initializer=tf.contrib.layers.xavier_initializer(),
                                  constraint=lambda x: tf.clip_by_value(x, 0, np.infty))

        self.W2 = tf.get_variable('Weights_NoConst', shape=[self.nx_noCons, self.ny],
                                  initializer=tf.contrib.layers.xavier_initializer())

        self.b = tf.get_variable('Biases', shape=[1, self.ny], initializer=tf.constant_initializer())

        # Model
        self.z = tf.matmul(self.X1, self.W1) + tf.matmul(self.X2, self.W2) + self.b

        # Mean squared error loss function
        self.regularizer = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)
        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.z) +
                                   self.lambd * self.regularizer)

        # Optimization, Adaptive Momentum Gradient Descent
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

        # Initializations & saving
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        # Loss history
        self.loss_history = []

    def __str__(self):
        return "Linear Regression using {} features.".format(self.nx_Cons + self.nx_noCons)

    def __repr__(self):
        return "LinearRegression()"

    def train(self, const_features, unconst_features, labels):
        """
        Description
           ---
              Train linear regression.  Runs the optimizer on the loss function


        Inputs
           ---
              features: Features of the machine learning model (X)
                labels: Targets / Labels of the machine learning model (y)

        """

        _ = self.sess.run(self.optimizer, feed_dict={self.y: labels,
                                                     self.X1: const_features,
                                                     self.X2: unconst_features})

    def test(self, const_features, unconst_features):
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

        return self.sess.run(self.z, feed_dict={self.X1: const_features,
                                                self.X2: unconst_features})

    def loss_check(self, const_features, unconst_features, labels):
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

        cur_loss = self.sess.run(self.loss, feed_dict={self.y: labels,
                                                       self.X1: const_features,
                                                       self.X2: unconst_features})
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

        return self.sess.run([self.W1, self.W2, self.b])

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


def simulation(data_path, model_path, norm_path, test_size=0.05, shuffle=True, lr=0.003, minibatch_size=2048,
               train_size=0.9, epochs=30, lambd=0.001, testing=False, num_of_const=10):
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

    raw_data = pd.read_csv(data_path)

    heading_names = list(raw_data)
    raw_data = raw_data.values

    print('There are {} feature(s) and {} label(s) with {} examples.'.format(raw_data.shape[1] - 1, 1,
                                                                             raw_data.shape[0]))

    # Train / Test split
    train_x, test_x, train_y, test_y = train_test_split(raw_data[:, 1:], raw_data[:, 0],
                                                        test_size=test_size, shuffle=shuffle, random_state=42)

    # Reshape for TensorFlow
    train_x = train_x.reshape(-1, raw_data.shape[1] - 1)
    test_x = test_x.reshape(-1, raw_data.shape[1] - 1)

    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    # Test cases for NaN values
    assert(not np.isnan(train_x).any())
    assert(not np.isnan(test_x).any())

    assert(not np.isnan(train_y).any())
    assert(not np.isnan(test_y).any())

    with tf.Session() as sess:

        # Build linear regression object
        linear_reg = LinearRegression(sess, train_x, train_y, test_x, test_y, lr=lr, minibatch_size=minibatch_size,
                                      train_size=train_size, epochs=epochs, lambd=lambd, num_of_const=num_of_const)

        # If testing, just run it
        if testing:
            # Restore model
            linear_reg.saver.restore(sess, save_path=model_path)

            # Pred testing values.  First num_of_const variables are constrained, remaining are unconstrained
            pred = linear_reg.test(test_x[:, :num_of_const], test_x[:, linear_reg.nx_Cons:])

            # Evaluate loss
            rmse, mae = linear_reg.eval_loss(pred, test_y)

            print('Test RMSE: {:2f} | Test MAE: {:2f}'.format(rmse, mae))

            weights_biases = linear_reg.weights_and_biases()

        else:
            # Global variables initializer
            sess.run(linear_reg.init)

            for epoch in range(linear_reg.epochs):

                for i in range(linear_reg.total_batch_number):

                    # Mini-batch gradient descent
                    batch_index = i * linear_reg.minibatch_size
                    minibatch_x = train_x[batch_index:batch_index + linear_reg.minibatch_size, :]
                    minibatch_y = train_y[batch_index:batch_index + linear_reg.minibatch_size, :]

                    # Optimize machine learning model
                    linear_reg.train(const_features=minibatch_x[:, :num_of_const],
                                     unconst_features=minibatch_x[:, linear_reg.nx_Cons:],
                                     labels=minibatch_y)

                    # Record loss
                    if i % 10 == 0:
                        _ = linear_reg.loss_check(const_features=train_x[:, :num_of_const],
                                                  unconst_features=train_x[:, linear_reg.nx_Cons:],
                                                  labels=train_y)

                    # Evaluate train and test losses
                    if i % 150 == 0:
                        current_loss = linear_reg.loss_check(const_features=train_x[:, :num_of_const],
                                                             unconst_features=train_x[:, linear_reg.nx_Cons:],
                                                             labels=train_y)

                        train_pred = linear_reg.test(const_features=train_x[:, :num_of_const],
                                                     unconst_features=train_x[:, linear_reg.nx_Cons:])

                        # Evaluate error
                        train_rmse, train_mae = linear_reg.eval_loss(train_pred, train_y)

                        test_pred = linear_reg.test(const_features=test_x[:, :num_of_const],
                                                    unconst_features=test_x[:, linear_reg.nx_Cons:])

                        test_rmse, test_mae = linear_reg.eval_loss(test_pred, test_y)

                        print('Epoch: {} | Loss: {:2f} | Train RMSE: {:2f} | Test RMSE: {:2f}'.format(epoch,
                                                                                                      current_loss,
                                                                                                      train_rmse,
                                                                                                      test_rmse))

            # Save model
            linear_reg.saver.save(sess, model_path)
            print("Model saved at: {}".format(model_path))

            # Final test
            test_pred = linear_reg.test(const_features=test_x[:, :num_of_const],
                                        unconst_features=test_x[:, linear_reg.nx_Cons:])

            test_rmse, test_mae = linear_reg.eval_loss(test_pred, test_y)
            print('Final Test Results:  Test RMSE: {:2f} | Test MAE: {:2f}'.format(test_rmse, test_mae))

            weights_biases = linear_reg.weights_and_biases()

    return raw_data, heading_names, linear_reg, weights_biases


if __name__ == "__main__":

    # Seed NumPy and TensorFlow with the meaning of life
    random_seed(42)

    # Specify data, model and normalization paths
    Data_path = '/home/rui/Documents/Willowglen/data/Optimization_Data/Opti_withAllChangable.csv'
    Model_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/linear_regression' \
                 '_models/Objects/checkpoints/ls_noNorm.ckpt'
    Norm_path = '/home/rui/Documents/Willowglen/Suncor_Phase2/linear_regression' \
                '_models/Objects/normalization/ls_noNorm.pickle'

    Raw_data, Heading_names, Linear_reg, Weights_biases = simulation(Data_path, Model_path, Norm_path, test_size=0.05,
                                                                     shuffle=True, lr=0.003, minibatch_size=2048,
                                                                     train_size=0.9, epochs=4000, lambd=0.001,
                                                                     testing=False, num_of_const=10)
