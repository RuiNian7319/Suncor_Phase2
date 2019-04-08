"""
Linear Regression Patch 1.0

Patch notes:  Added tensorboard, saver

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
path = '/home/rui/Documents/Willowglen/data/PresCur_Data/'

# Arguments
parser.add_argument("--data", help="Data to be loaded into the model", default=path + 'Cur2Pres_Chey.csv')
parser.add_argument("--train_size", help="% of whole data set used for training", default=0.95)
parser.add_argument('--lr', help="learning rate for the linear regression", default=0.003)
parser.add_argument("--minibatch_size", help="mini batch size for mini batch gradient descent", default=1024)
parser.add_argument("--epochs", help="Number of times data should be recycled through", default=2000)
parser.add_argument("--tensorboard_path", help="Location of saved tensorboard information", default="./tensorboard")
parser.add_argument("--model_path", help="Location of saved tensorflow graph", default='checkpoints/Chey_noNorm.ckpt')
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

# Model placeholders
with tf.name_scope("Inputs"):
    x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
    y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# Model variables
with tf.name_scope("Model"):
    with tf.variable_scope("Weights"):
        W = tf.get_variable('Weights', shape=[input_size, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('Biases', shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())

    tf.summary.histogram("Weights", W)
    tf.summary.histogram("Biases", b)

# Model
z = tf.matmul(x, W) + b

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
            minibatch_train_X = train_X[batch_index:(batch_index + mini_batch_size), :]
            minibatch_train_y = train_y[batch_index:(batch_index + mini_batch_size), :]

            _ = sess.run(optimizer, feed_dict={x: minibatch_train_X, y: minibatch_train_y})
            # summary = sess.run(merge, feed_dict={x: minibatch_train_X, y: minibatch_train_y})
            current_loss = sess.run(loss, feed_dict={x: minibatch_train_X, y: minibatch_train_y})

            loss_history.append(current_loss)

            # Evaluate losses
            if i % 550 == 0:

                # # Add to summary writer
                # summary_writer.add_summary(summary, i)

                """
                Train data
                """

                train_pred = sess.run(z, feed_dict={x: train_X, y: train_y})
                train_loss = np.sqrt(np.mean(np.square(np.subtract(train_y, train_pred))))

                """
                Test data
                """

                test_pred = sess.run(z, feed_dict={x: test_X, y: test_y})
                test_loss = np.sqrt(np.mean(np.square(np.subtract(test_y, test_pred))))

                print('Epoch: {} | Loss: {:2f} | Train Error: {:2f} | Test Error: {:2f}'.format(epoch,
                                                                                                current_loss,
                                                                                                train_loss,
                                                                                                test_loss))

    if Args['save_graph']:
        save_path = saver.save(sess, Args["model_path"])
        print("Model was saved in {}".format(save_path))

    # Output weights
    weights = sess.run(W)
    biases = sess.run(b)

    # Predictions
    predictions = sess.run(z, feed_dict={x: test_X, y: test_y})

    # RMSE & MAE Calc
    RMSE_loss = np.sqrt(np.mean(np.square(np.subtract(test_y, predictions))))
    MAE_loss = np.mean(np.abs(np.subtract(test_y, predictions)))

    print('RMSE: {} | MAE: {}'.format(RMSE_loss, MAE_loss))

    # Visualization of what it looks like
    seq_pred(session=sess, model=z, features=x, normalizer=None, data=raw_data, time_start=1, time_end=5000,
             adv_plot=False, xlabel='Time', ylabel='Discharge Pressure')
