"""
Artificial Neural Network Regression Code v1.0 (Feed-forward neural network)

By: Rui Nian

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

import seaborn as sns
import gc

from copy import deepcopy

import warnings
import os

import sys
sys.path.insert(0, '/home/rui/Documents/Willowglen/Suncor_Phase2')

from EWMA import ewma
from Seq_plot import seq_pred
from MinMaxNorm import MinMaxNormalization

sns.set()
sns.set_style('white')

gc.enable()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

# Load data
# path = '/Users/ruinian/Documents/Willowglen/data/Optimization_Data/'
path = '/home/rui/Documents/Willowglen/data/Optimization_Data/'

raw_data = pd.read_csv(path + 'Opti_withAllChangableDenCurv3.csv')

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

train_X = training_data[:, 1:].reshape(-1, raw_data.shape[1] - 1)
test_X = testing_data[:, 1:].reshape(-1, raw_data.shape[1] - 1)

train_y = training_data[:, 0].reshape(-1, 1)
test_y = testing_data[:, 0].reshape(-1, 1)

input_size = train_X.shape[1]
h1_nodes = 30
h2_nodes = 30
h3_nodes = 30
output_size = 1

batch_size = 512
total_batch_number = int(train_X.shape[0] / batch_size)

X = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# Batch normalization
training = True
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
lambd = 0.000
trainable_vars = tf.trainable_variables()
lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in trainable_vars if 'bias' not in v.name]) * lambd

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=output) + lossL2)

# Batch Normalization
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

init = tf.global_variables_initializer()

epochs = 2
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

                if i % 550 == 0:

                    """
                    Train data
                    """

                    train_pred = sess.run(output, feed_dict={X: train_X, y: train_y, is_train: training})

                    # Unnormalize data
                    train_pred = np.multiply(train_pred, min_max_normalization.denominator[0, 0])
                    train_pred = train_pred + min_max_normalization.col_min[0, 0]

                    actual_labels = np.multiply(train_y, min_max_normalization.denominator[0, 0])
                    actual_labels = actual_labels + min_max_normalization.col_min[0, 0]

                    train_loss = np.sqrt(np.mean(np.square(np.subtract(actual_labels, train_pred))))

                    """
                    Test data
                    """

                    test_pred = sess.run(output, feed_dict={X: test_X, y: test_y, is_train: training})

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

    else:
        saver.restore(sess, 'checkpoints/pipeline_model.ckpt')

    saver.save(sess, 'checkpoints/pipeline_model.ckpt')

    predictions = sess.run(output, feed_dict={X: test_X, y: test_y, is_train: training})

    # Unnormalize data
    predictions = np.multiply(predictions, min_max_normalization.denominator[0, 0])
    predictions = predictions + min_max_normalization.col_min[0, 0]

    test_y = np.multiply(test_y, min_max_normalization.denominator[0, 0])
    test_y = test_y + min_max_normalization.col_min[0, 0]

    # RMSE & MAE Calc
    RMSE_loss = np.sqrt(np.mean(np.square(np.subtract(test_y, predictions))))
    MAE_loss = np.mean(np.abs(np.subtract(test_y, predictions)))

    print('RMSE: {} | MAE: {}'.format(RMSE_loss, MAE_loss))

    # Non-scrambled data plot
    # seq_pred(sess, output, raw_data, min_max_normalization, 0, 5000, adv_plot=False)

    # Pickle normalization
    pickle_out = open('normalization/norm_nn.pickle', 'wb')
    pickle.dump(min_max_normalization, pickle_out)
    pickle_out.close()
