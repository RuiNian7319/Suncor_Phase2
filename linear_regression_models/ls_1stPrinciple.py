"""
1st Principle pipeline model Patch 1.1

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

"""
Parsing section, to define parameters to be ran in the code
"""

# Initiate the parser
parser = argparse.ArgumentParser(description="Inputs to the linear regression")

# MacOS & Ubuntu 18.04 path
# path = '/Users/ruinian/Documents/Willowglen/data/'
path = '/home/rui/Documents/Willowglen/data/Optimization_Data/'

# Arguments
parser.add_argument("--data", help="Data to be loaded into the model", default=path + 'Opti_1stPrinc.csv')
parser.add_argument("--train_size", help="% of whole data set used for training", default=0.95)
parser.add_argument('--lr', help="learning rate for the linear regression", default=0.003)
parser.add_argument("--minibatch_size", help="mini batch size for mini batch gradient descent", default=4096)
parser.add_argument("--epochs", help="Number of times data should be recycled through", default=1000)
parser.add_argument("--tensorboard_path", help="Location of saved tensorboard information", default="./tensorboard")
parser.add_argument("--model_path", help="Location of saved tensorflow graph", default='checkpoints/ls_1stPrinc.ckpt')
parser.add_argument("--save_graph", help="Save the current tensorflow computational graph", default=True)
parser.add_argument("--restore_graph", help="Reload model parameters from saved location", default=True)

# Test Model
parser.add_argument("--test", help="put as true if you want to test the current model", default=False)

# Makes a dictionary of parsed args
Args = vars(parser.parse_args())

# Seed for reproducability
seed = 18
np.random.seed(seed)
tf.set_random_seed(seed)

# Loading data
raw_data = pd.read_csv(Args['data'])

# Turn Pandas dataframe into NumPy Array
raw_data = raw_data.values
print("Raw data has {} features with {} examples.".format(raw_data.shape[1], raw_data.shape[0]))

train_X, test_X, train_y, test_y = train_test_split(raw_data[:, 1:], raw_data[:, 0],
                                                    test_size=0.05, random_state=42, shuffle=True)

train_X = train_X.reshape(-1, raw_data.shape[1] - 1)
test_X = test_X.reshape(-1, raw_data.shape[1] - 1)

train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

# Normalization.  Recombine to normalize at once, then split them into their train/test forms
min_max_normalization = MinMaxNormalization(np.concatenate([train_y, train_X], axis=1))
training_data = min_max_normalization(np.concatenate([train_y, train_X], axis=1))
testing_data = min_max_normalization(np.concatenate([test_y, test_X], axis=1))

# Data partitioning for 1st principle model
# DRA
train_X1 = training_data[:, 1:3].reshape(-1, 2)
test_X1 = testing_data[:, 1:3].reshape(-1, 2)
DRA_size = 2

# VFD
train_X2 = training_data[:, 3:5].reshape(-1, 2)
test_X2 = testing_data[:, 3:5].reshape(-1, 2)
VFD_size = 2

# Pump
train_X3 = training_data[:, 5:].reshape(-1, 4)
test_X3 = testing_data[:, 5:].reshape(-1, 4)
Pump_size = 4

train_y = training_data[:, 0].reshape(-1, 1)
test_y = testing_data[:, 0].reshape(-1, 1)

# Neural network parameters
input_size = train_X.shape[1]
output_size = 1
learning_rate = Args['lr']
mini_batch_size = Args['minibatch_size']
total_batch_number = int(train_X.shape[0] / mini_batch_size)
epochs = Args['epochs']

# Test cases
assert(not np.isnan(train_X).any())
assert(not np.isnan(test_X).any())

# Model constants
with tf.name_scope("Constants"):
    gravity = tf.constant(9.81, dtype=tf.float32, shape=[1, 1])
    head = tf.constant(152, dtype=tf.float32, shape=[1, 1])
    convert = tf.constant(6.29, dtype=tf.float32, shape=[1, 1])

# Model placeholders
with tf.name_scope("Inputs"):
    X_vfd = tf.placeholder(dtype=tf.float32, shape=[None, VFD_size])
    X_pump = tf.placeholder(dtype=tf.float32, shape=[None, Pump_size])
    X_dra = tf.placeholder(dtype=tf.float32, shape=[None, DRA_size])
    y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# Model variables
with tf.name_scope("Model"):
    with tf.variable_scope("Weights"):
        W_vfd = tf.get_variable('VFD_W', shape=[VFD_size, 1], initializer=tf.contrib.layers.xavier_initializer())
        W_pump = tf.get_variable('Pump_W', shape=[Pump_size, 1], initializer=tf.contrib.layers.xavier_initializer())
        W_U = tf.get_variable('U_W', shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())
        W_dra = tf.get_variable('DRA_W', shape=[DRA_size, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('Biases', shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())

    # tf.summary.histogram("VFD_W", W_vfd)
    # tf.summary.histogram("Pump_W", W_pump)
    # tf.summary.histogram("Internal_Energy W", W_U)
    # tf.summary.histogram("DRA_W", W_dra)
    # tf.summary.histogram("Biases", b)

# 1st Principle Model
model = tf.divide(tf.matmul(X_vfd, W_vfd) + tf.matmul(X_pump, W_pump),
                  W_U + gravity * head) + tf.matmul(X_dra, W_dra) + b
z = tf.multiply(model, convert)

# Cross entropy with logits, assumes inputs are logits before cross entropy
loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=z))

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

loss_history = []

# Initialize all variables
init = tf.global_variables_initializer()

# Tensorflow graph saver
saver = tf.train.Saver()

with tf.Session() as sess:

    # summary_writer = tf.summary.FileWriter(Args['tensorboard_path'], graph=sess.graph)
    # merge = tf.summary.merge_all()

    if Args['restore_graph']:
        # Restore tensorflow graph
        saver.restore(sess, Args['model_path'])
    else:
        sess.run(init)

    for epoch in range(epochs):

        for i in range(total_batch_number):

            # Generate mini_batch indexing
            batch_index = i * mini_batch_size
            minibatch_train_X1 = train_X1[batch_index:(batch_index + mini_batch_size), :]
            minibatch_train_X2 = train_X2[batch_index:(batch_index + mini_batch_size), :]
            minibatch_train_X3 = train_X3[batch_index:(batch_index + mini_batch_size), :]

            minibatch_train_y = train_y[batch_index:(batch_index + mini_batch_size), :]

            _ = sess.run(optimizer, feed_dict={X_dra: minibatch_train_X1,
                                               X_vfd: minibatch_train_X2,
                                               X_pump: minibatch_train_X3,
                                               y: minibatch_train_y})

            # summary = sess.run(merge, feed_dict={X_dra: minibatch_train_X1,
            #                                      X_vfd: minibatch_train_X2,
            #                                      X_pump: minibatch_train_X3,
            #                                      y: minibatch_train_y})

            current_loss = sess.run(loss, feed_dict={X_dra: minibatch_train_X1,
                                                     X_vfd: minibatch_train_X2,
                                                     X_pump: minibatch_train_X3,
                                                     y: minibatch_train_y})

            loss_history.append(current_loss)

            # Evaluate losses
            if i % 550 == 0:

                # Add to summary writer
                # summary_writer.add_summary(summary, i)

                """
                Train data
                """

                train_pred = sess.run(z, feed_dict={X_dra: train_X1,
                                                    X_vfd: train_X2,
                                                    X_pump: train_X3,
                                                    y: train_y})

                # Unnormalize data
                train_pred = np.multiply(train_pred, min_max_normalization.denominator[0, 0])
                train_pred = train_pred + min_max_normalization.col_min[0, 0]

                actual_labels = np.multiply(train_y, min_max_normalization.denominator[0, 0])
                actual_labels = actual_labels + min_max_normalization.col_min[0, 0]

                train_loss = np.sqrt(np.mean(np.square(np.subtract(actual_labels, train_pred))))

                """
                Test data
                """

                test_pred = sess.run(z, feed_dict={X_dra: test_X1,
                                                   X_vfd: test_X2,
                                                   X_pump: test_X3,
                                                   y: test_y})

                # Unnormalize data
                test_pred = np.multiply(test_pred, min_max_normalization.denominator[0, 0])
                test_pred = test_pred + min_max_normalization.col_min[0, 0]

                actual_labels = np.multiply(test_y, min_max_normalization.denominator[0, 0])
                actual_labels = actual_labels + min_max_normalization.col_min[0, 0]

                test_loss = np.sqrt(np.mean(np.square(np.subtract(actual_labels, test_pred))))

                print('Epoch: {} | Loss: {:2f} | Train Error: {:2f} | Test Error: {:2f}'.format(epoch,
                                                                                                current_loss,
                                                                                                train_loss,
                                                                                                test_loss))

    if Args['save_graph']:
        save_path = saver.save(sess, Args["model_path"])
        print("Model was saved in {}".format(save_path))

    # Output weights
    weights_dra = sess.run(W_dra)
    weights_vfd = sess.run(W_vfd)
    weights_pump = sess.run(W_pump)
    weights_U = sess.run(W_U)
    biases = sess.run(b)

    # Predictions
    predictions = sess.run(z, feed_dict={X_dra: test_X1,
                                         X_vfd: test_X2,
                                         X_pump: test_X3,
                                         y: test_y})

    # Unnormalize data
    predictions = np.multiply(predictions, min_max_normalization.denominator[0, 0])
    predictions = predictions + min_max_normalization.col_min[0, 0]

    test_y = np.multiply(test_y, min_max_normalization.denominator[0, 0])
    test_y = test_y + min_max_normalization.col_min[0, 0]

    # RMSE & MAE Calc
    RMSE_loss = np.sqrt(np.mean(np.square(np.subtract(test_y, predictions))))
    MAE_loss = np.mean(np.abs(np.subtract(test_y, predictions)))

    print('RMSE: {} | MAE: {}'.format(RMSE_loss, MAE_loss))

    # Pickle normalization
    pickle_out = open('normalization/ls.pickle', 'wb')
    pickle.dump(min_max_normalization, pickle_out)
    pickle_out.close()
