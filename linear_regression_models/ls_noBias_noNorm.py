"""
Linear Regression Patch 1.0

Patch notes:  Added tensorboard, saver

Date of last edit: February 18th
Rui Nian

Current issues: Output size is hard coded
                Cannot run code purely to test the accuracy of algorithm
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

gc.enable()

"""
Parsing section, to define parameters to be ran in the code
"""

# Initiate the parser
parser = argparse.ArgumentParser(description="Inputs to the linear regression")
path = '/Users/ruinian/Documents/Willowglen/data/'

# Arguments
parser.add_argument("--data", help="Data to be loaded into the model", default=path + 'flow_nn_data_BoosterPump.csv')
parser.add_argument("--train_size", help="% of whole data set used for training", default=0.95)
parser.add_argument('--lr', help="learning rate for the logistic regression", default=0.003)
parser.add_argument("--minibatch_size", help="mini batch size for mini batch gradient descent", default=512)
parser.add_argument("--epochs", help="Number of times data should be recycled through", default=35)
parser.add_argument("--tensorboard_path", help="Location of saved tensorboard information", default="./tensorboard")
parser.add_argument("--model_path", help="Location of saved tensorflow graph", default='checkpoints/time1.ckpt')
parser.add_argument("--save_graph", help="Save the current tensorflow computational graph", default=True)
parser.add_argument("--restore_graph", help="Reload model parameters from saved location", default=False)

# Test Model
parser.add_argument("--test", help="put as true if you want to test the current model", default=False)

# Makes a dictionary of parsed args
Args = vars(parser.parse_args())

"""
Logistic Regression
"""

# Seed for reproducability
seed = 18
np.random.seed(seed)
tf.set_random_seed(seed)


# Min max normalization
class MinMaxNormalization:
    """
    Inputs
       -----
            data:  Input feature vectors from the training data
    Attributes
       -----
         col_min:  The minimum value per feature
         col_max:  The maximum value per feature
     denominator:  col_max - col_min
     Methods
        -----
     init:  Builds the col_min, col_max, and denominator
     call:  Normalizes data based on init attributes
    """

    def __init__(self, data):
        self.col_min = np.min(data, axis=0).reshape(1, data.shape[1])
        self.col_max = np.max(data, axis=0).reshape(1, data.shape[1])
        self.denominator = abs(self.col_max - self.col_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.denominator[0]):
            if value == 0:
                self.denominator[0][index] = 1

    def __call__(self, data):
        return np.divide((data - self.col_min), self.denominator)

    def unnormalize(self, data):

        data = np.multiply(data, self.denominator)
        data = data + self.col_min

        return data


# Loading data
raw_data = pd.read_csv(Args['data'])

# Turn Pandas dataframe into NumPy Array
raw_data = raw_data.values
print("Raw data has {} features with {} examples.".format(raw_data.shape[1], raw_data.shape[0]))

train_X, test_X, train_y, test_y = train_test_split(raw_data[:, 1:], raw_data[:, 0],
                                                    test_size=0.05, random_state=42, shuffle=True)

train_X = train_X.reshape(-1, 7)
test_X = test_X.reshape(-1, 7)

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
assert(np.isnan(train_X).any() == False)
assert(np.isnan(test_X).any() == False)

# Model placeholders
with tf.name_scope("Inputs"):
    x = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
    y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# Model variables
with tf.name_scope("Model"):
    with tf.variable_scope("Weights"):
        W = tf.get_variable('Weights', shape=[input_size, 1], initializer=tf.contrib.layers.xavier_initializer())

    tf.summary.histogram("Weights", W)

# Model
z = tf.matmul(x, W)

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

    summary_writer = tf.summary.FileWriter(Args['tensorboard_path'], graph=sess.graph)
    merge = tf.summary.merge_all()

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

            _, summary = sess.run([optimizer, merge], feed_dict={x: minibatch_train_X, y: minibatch_train_y})
            current_loss = sess.run(loss, feed_dict={x: minibatch_train_X, y: minibatch_train_y})

            loss_history.append(current_loss)

            if i % 550 == 0:

                # Add to summary writer
                summary_writer.add_summary(summary, i)

                """
                Train data
                """

                train_pred = sess.run(z, feed_dict={x: train_X, y: train_y})

                train_loss = np.sqrt(np.median(np.square(np.subtract(train_y, train_pred))))

                """
                Test data
                """

                test_pred = sess.run(z, feed_dict={x: test_X, y: test_y})

                test_loss = np.sqrt(np.median(np.square(np.subtract(test_y, test_pred))))

                print('Epoch: {} | Loss: {:2f} | Train Error: {:2f} | Test Error: {:2f}'.format(epoch,
                                                                                                current_loss,
                                                                                                train_loss,
                                                                                                test_loss))

    if Args['save_graph']:
        save_path = saver.save(sess, Args["model_path"])
        print("Model was saved in {}".format(save_path))

    # Output weights

    weights = sess.run(W)

    # Predictions
    predictions = sess.run(z, feed_dict={x: test_X})


    # MSE Calc
    test_loss = np.sqrt(np.mean(np.square(np.subtract(test_y, predictions))))
    print(test_loss)

    # Visualization of what it looks like
    plt.plot(test_y[100:150], color='grey')
    plt.plot(predictions[100:150], color='green')

    plt.xlabel('Samples')
    plt.ylabel('Flow rate, bbl/h')
    plt.show()