"""
Linear Regression Patch 1.1

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
parser.add_argument("--data", help="Data to be loaded into the model", default=path + 'Opti_withAllChangableDenCurv3.csv')
parser.add_argument("--train_size", help="% of whole data set used for training", default=0.95)
parser.add_argument('--lr', help="learning rate for the linear regression", default=0.003)
parser.add_argument("--minibatch_size", help="mini batch size for mini batch gradient descent", default=1024)
parser.add_argument("--epochs", help="Number of times data should be recycled through", default=200)
parser.add_argument("--tensorboard_path", help="Location of saved tensorboard information", default="./tensorboard")
parser.add_argument("--model_path", help="Location of saved tensorflow graph", default='checkpoints/ls.ckpt')
parser.add_argument("--save_graph", help="Save the current tensorflow computational graph", default=True)
parser.add_argument("--restore_graph", help="Reload model parameters from saved location", default=False)

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

# X1: Constrained, strictly positive weights  X2: Unconstrained weights
train_X1 = train_X[:, :10].reshape(-1, 10)
train_X2 = train_X[:, 10:].reshape(-1, 3)

test_X1 = test_X[:, :10].reshape(-1, 10)
test_X2 = test_X[:, 10:].reshape(-1, 3)

train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

# Neural network parameters
input_size1 = train_X1.shape[1]
input_size2 = train_X2.shape[1]
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
    X1 = tf.placeholder(dtype=tf.float32, shape=[None, input_size1])
    X2 = tf.placeholder(dtype=tf.float32, shape=[None, input_size2])
    y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# Model variables
with tf.name_scope("Model"):
    with tf.variable_scope("Weights"):
        W1 = tf.get_variable('Weights_Cons', shape=[input_size1, 1], initializer=tf.contrib.layers.xavier_initializer(),
                             constraint=lambda x: tf.clip_by_value(x, 0, np.infty))
        W2 = tf.get_variable('Weights_Norm', shape=[input_size2, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('Biases', shape=[1, 1], initializer=tf.contrib.layers.xavier_initializer())

    # tf.summary.histogram("Weights", W1)
    # tf.summary.histogram("Weights", W2)
    # tf.summary.histogram("Biases", b)

# Model
z = tf.matmul(X1, W1) + tf.matmul(X2, W2) + b

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
            minibatch_train_y = train_y[batch_index:(batch_index + mini_batch_size), :]

            _ = sess.run(optimizer, feed_dict={X1: minibatch_train_X1, X2: minibatch_train_X2, y: minibatch_train_y})
            # summary = sess.run(merge, feed_dict={X1: minibatch_train_X, X2: minibatch_train_X2, y: minibatch_train_y})
            current_loss = sess.run(loss, feed_dict={X1: minibatch_train_X1, X2: minibatch_train_X2,
                                                     y: minibatch_train_y})

            loss_history.append(current_loss)

            # Evaluate losses
            if i % 550 == 0:

                # Add to summary writer
                # summary_writer.add_summary(summary, i)

                """
                Train data
                """

                train_pred = sess.run(z, feed_dict={X1: train_X1, X2: train_X2, y: train_y})

                train_loss = np.sqrt(np.mean(np.square(np.subtract(train_y, train_pred))))

                """
                Test data
                """

                test_pred = sess.run(z, feed_dict={X1: test_X1, X2: test_X2, y: test_y})

                test_loss = np.sqrt(np.mean(np.square(np.subtract(test_y, test_pred))))

                print('Epoch: {} | Loss: {:2f} | Train Error: {:2f} | Test Error: {:2f}'.format(epoch,
                                                                                                current_loss,
                                                                                                train_loss,
                                                                                                test_loss))

    if Args['save_graph']:
        save_path = saver.save(sess, Args["model_path"])
        print("Model was saved in {}".format(save_path))

    # Output weights
    weights1 = sess.run(W1)
    weights2 = sess.run(W2)
    biases = sess.run(b)

    # Predictions
    predictions = sess.run(z, feed_dict={X1: test_X1, X2: test_X2, y: test_y})

    # RMSE & MAE Calc
    RMSE_loss = np.sqrt(np.mean(np.square(np.subtract(test_y, predictions))))
    MAE_loss = np.mean(np.abs(np.subtract(test_y, predictions)))

    print('RMSE: {} | MAE: {}'.format(RMSE_loss, MAE_loss))

    """
    Plotting LoL
    """

    time_start = 1
    time_end = 3000

    # Reshape the data for tensorflow
    plot_x1 = raw_data[time_start:time_end, 1:11].reshape(-1, 10)
    plot_x2 = raw_data[time_start:time_end, 11:].reshape(-1, 3)
    plot_y = raw_data[time_start:time_end, 0].reshape(-1, 1)

    # Run tensorflow model to get predicted y
    preds = sess.run(z, feed_dict={X1: plot_x1, X2: plot_x2})

    # RMSE & MAE Calc
    rmse_loss = np.sqrt(np.mean(np.square(np.subtract(plot_y, preds))))
    mae_loss = np.mean(np.abs(np.subtract(plot_y, preds)))

    print('Seq. RMSE: {} | Seq. MAE: {}'.format(rmse_loss, mae_loss))

    # Visualization of model in production with fancy error bars
    stderr = np.std(np.abs(np.subtract(plot_y, preds)))

    group1 = np.concatenate([np.linspace(0, time_end - time_start - 1, time_end - time_start).reshape(-1, 1),
                             preds[0:time_end - time_start]], axis=1)
    group2 = np.concatenate([np.linspace(0, time_end - time_start - 1, time_end - time_start).reshape(-1, 1),
                             preds[0:time_end - time_start] + stderr], axis=1)
    group3 = np.concatenate([np.linspace(0, time_end - time_start - 1, time_end - time_start).reshape(-1, 1),
                             preds[0:time_end - time_start] - stderr], axis=1)

    group = np.concatenate([group1, group2, group3])

    df = pd.DataFrame(group, columns=['time', 'predictions'])

    sns.lineplot(x='time', y='predictions', data=df)
    plt.plot(plot_y[time_start:time_end], label='Actual')

    plt.xlabel('Time')
    plt.ylabel('Flow rate')
    plt.legend(loc=0, frameon=False)

    plt.show()

