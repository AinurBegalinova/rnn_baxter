import sys
import tensorflow as tf 
import pandas as pd 
import numpy as np 
import os
import matplotlib
import matplotlib.pyplot as plt 
import random
import shutil
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics 
import tensorflow.contrib.rnn as rnn 
from sklearn.utils import shuffle

# Helper class to perform K-Folds Validation splitting
class CrossValidationFolds(object):
    
    def __init__(self, data, labels, num_folds, shuffle=True):
        self.data = data
        self.labels = labels
        self.num_folds = num_folds
        self.current_fold = 0
        
        # Shuffle Dataset
        if shuffle:
            perm = np.random.permutation(self.data.shape[0])
            data = data[perm]
            labels = labels[perm]

    
    def split(self):
        current = self.current_fold
        size = int(self.data.shape[0]/self.num_folds)
        
        index = np.arange(self.data.shape[0])
        lower_bound = index >= current*size
        upper_bound = index < (current + 1)*size
        cv_region = lower_bound*upper_bound

        cv_data = self.data[cv_region]
        train_data = self.data[~cv_region]
        
        cv_labels = self.labels[cv_region]
        train_labels = self.labels[~cv_region]
        
        self.current_fold += 1
        return (train_data, train_labels), (cv_data, cv_labels), (size, cv_region)



PATH = 'DATASET/'
TRAIN = 'traj_1_train_shuffled.csv'
TEST = 'traj_1_test_shuffled.csv'

print('Reading CSV Data...')
df = pd.read_csv(PATH + TRAIN)
test_df = pd.read_csv(PATH + TEST)

#Create a new feature for normal (non-fraudulent) transactions.
df.loc[df.label == 0, 'Slip'] = 1
df.loc[df.label == 1, 'Slip'] = 0

test_df.loc[test_df.label == 0, 'Slip'] = 1
test_df.loc[test_df.label == 1, 'Slip'] = 0

#Rename 'Class' to 'Fraud'.
df = df.rename(columns={'label': 'Stable'})
test_df = test_df.rename(columns={'label': 'Stable'})

#Create dataframes of only Fraud and Normal transactions.
Slip = df[df.Slip == 1]
Stable = df[df.Stable == 1]

test_Slip = test_df[test_df.Slip == 1]
test_Stable = test_df[test_df.Stable == 1]

# Set X_train equal to 80% of the fraudulent transactions.
#X_train = Slip.sample
#count_Slips = len(Slip)

# Add 80% of the normal transactions to X_train.
X_train = pd.concat([Slip, Stable], axis = 0)
X_test = pd.concat([test_Slip, test_Stable], axis = 0)

# X_test contains all the transaction not in X_train.
#X_test = df.loc[~df.index.isin(X_train.index)]

#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(test_df)

#Add our target features to y_train and y_test.
y_train = X_train.Slip
y_train = pd.concat([y_train, X_train.Stable], axis=1)

y_test = X_test.Slip
y_test = pd.concat([y_test, X_test.Stable], axis=1)

#Drop target features from X_train and X_test.
X_train = X_train.drop(['Slip','Stable'], axis = 1)
X_test = X_test.drop(['Slip','Stable'], axis = 1)


#Select certain features
#cols = [c for c in X_train.columns if c.lower()[:5] != 'median']
X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='left')))]
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='left')))]

X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='right')))]
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='right')))]

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


X_train = X_train[:, 3]
y_train = y_train[:, 1]

X_test = X_test[:, 3]
y_test = y_test [:, 1]
print (X_train.shape)
print (y_train.shape)


num_periods = 20
f_horizon = 1

X_train = X_train[:(len(X_train) -(len(X_train) % num_periods))]
x_batches = X_train.reshape(-1, 20, 1)

y_train = y_train[1:(len(y_train) -(len(y_train) % num_periods))+f_horizon]
y_batches = y_train.reshape(-1, 20, 1)
'''
print (len(x_batches))
print (x_batches.shape)
print (x_batches[0:2])

print (y_batches[0:1])
print (y_batches.shape)'''


X_test_setup = X_test[-(num_periods + f_horizon):]
X_test_temp = X_test_setup[:num_periods].reshape(-1, 20, 1)
y_test = X_test[-(num_periods):].reshape(-1, 20, 1)
X_test = X_test_temp

#print (X_test)
#print (X_test.shape)


tf.reset_default_graph()

num_periods = 20
inputs = 1
hidden = 100
output = 1

x = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, x, dtype=tf.float32)

learning_rate = 0.001


stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])


#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=y))
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 100

with tf.Session() as sess:
	init.run()
	for ep in range(epochs):
		sess.run(training_op, feed_dict = {x: x_batches, y: y_batches})
		if ep % 10 == 0:
			mse = loss.eval(feed_dict={x: x_batches, y: y_batches})
			print(ep, "\tMSE:", mse)

	y_pred = sess.run(outputs, feed_dict={x: X_test})
	print (y_pred)

