"""
Artificial Neural Network Regression Code v1.0 (Feed-forward neural network)

By: Rui Nian

Date of last edit: February 18th, 2019

Patch Notes: -

Known Issues: -

Features:
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import pickle

from sklearn.model_selection import train_test_split

import gc

from copy import deepcopy

import warnings
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 1
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


# Load data
path = '/Users/ruinian/Documents/Willowglen/data/'
# path = '/home/rui/Documents/logistic_regression_tf/'

raw_data = pd.read_csv(path + 'flow_nn_data.csv')
raw_data = raw_data.values

min_max_normalization = MinMaxNormalization(raw_data)
raw_data = min_max_normalization(raw_data)

assert(not np.isnan(raw_data).any())

train_X, test_X, train_y, test_y = train_test_split(raw_data[:, 1:], raw_data[:, 0],
                                                    test_size=0.05, random_state=42, shuffle=True)

train_X = train_X.reshape(-1, 6)
test_X = test_X.reshape(-1, 6)

train_y = train_y.reshape(-1, 1)
test_y = test_y.reshape(-1, 1)

input_size = train_X.shape[1]
h1_nodes = 30
h2_nodes = 30
h3_nodes = 30
output_size = 1

batch_size = 256
total_batch_number = int(train_X.shape[0] / batch_size)

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# Batch normalization
training = False
is_train = tf.placeholder(dtype=tf.bool, name='is_train')

hidden_layer_1 = {'weights': tf.get_variable('h1_weights', shape=[input_size, h1_nodes],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h1_biases', shape=[h1_nodes],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

hidden_layer_2 = {'weights': tf.get_variable('h2_weights', shape=[h1_nodes, h2_nodes],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h2_biases', shape=[h2_nodes],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

hidden_layer_3 = {'weights': tf.get_variable('h3_weights', shape=[h2_nodes, h3_nodes],
                                             initializer=tf.contrib.layers.variance_scaling_initializer()),
                  'biases': tf.get_variable('h3_biases', shape=[h3_nodes],
                                            initializer=tf.contrib.layers.variance_scaling_initializer())}

output_layer = {'weights': tf.get_variable('output_weights', shape=[h3_nodes, output_size],
                                           initializer=tf.contrib.layers.variance_scaling_initializer()),
                'biases': tf.get_variable('output_biases', shape=[output_size],
                                          initializer=tf.contrib.layers.variance_scaling_initializer())}

l1 = tf.add(tf.matmul(X, hidden_layer_1['weights']), hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)
l1 = tf.layers.batch_normalization(l1, training=is_train)

l2 = tf.add(tf.matmul(l1, hidden_layer_2['weights']), hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)
l2 = tf.layers.batch_normalization(l2, training=is_train)

l3 = tf.add(tf.matmul(l2, hidden_layer_3['weights']), hidden_layer_3['biases'])
l3 = tf.nn.relu(l3)
l3 = tf.layers.batch_normalization(l3, training=is_train)

output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

# L2 Regularization
lambd = 0.005
trainable_vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * lambd

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=output) + lossL2)

# Batch Normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()

epochs = 5
loss_history = []

saver = tf.train.Saver()

with tf.Session() as sess:

    if training:
        sess.run(init)

        for epoch in range(epochs):

            for i in range(total_batch_number):
                batch_index = i * batch_size
                batch_X = train_X[batch_index:batch_index + batch_size, :]
                batch_y = train_y[batch_index:batch_index + batch_size, :]

                sess.run(optimizer, feed_dict={X: batch_X, y: batch_y, is_train: training})
                current_loss = sess.run(loss, feed_dict={X: batch_X, y: batch_y, is_train: training})
                loss_history.append(current_loss)

                if i % 50 == 0:
                    print('Epoch: {} | Loss: {}'.format(epoch, current_loss))

    else:
        saver.restore(sess, 'checkpoints/pipeline_model.ckpt')

    saver.save(sess, 'checkpoints/pipeline_model.ckpt')

    predictions = sess.run(output, feed_dict={X: test_X, y: test_y, is_train: training})

    # Unnormalize data
    predictions = np.multiply(predictions, min_max_normalization.denominator[0, 0])
    predictions = predictions + min_max_normalization.col_min[0, 0]

    test_y = np.multiply(test_y, min_max_normalization.denominator[0, 0])
    test_y = test_y + min_max_normalization.col_min[0, 0]

    # MSE Calc
    test_loss = np.sqrt(np.mean(np.square(np.subtract(test_y, predictions))))
    print(test_loss)

    # Visualization of what it looks like
    plt.plot(test_y[0:50], color='grey')
    plt.plot(predictions[0:50], color='green')
    plt.show()
