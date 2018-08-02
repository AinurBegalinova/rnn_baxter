
# coding: utf-8

# In[3]:


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



# In[4]:


data = 'data.csv'


# In[26]:


print('Reading CSV Data...')
df = pd.read_csv(data)

#Number of positive and negative classes
#print (df.label[df.label == 0].count())
#Create a new feature for normal (non-fraudulent) transactions.

#df.loc[df.label == 0, 'Slip'] = 1
#df.loc[df.label == 1, 'Slip'] = 0



#Rename 'Class' to 'Fraud'.
#df = df.rename(columns={'label': 'Stable'})
#test_df = test_df.rename(columns={'label': 'Stable'})

#Create dataframes of only Fraud and Normal transactions.
#Slip = df[df.Slip == 1]
#Stable = df[df.Stable == 1]

#test_Slip = test_df[test_df.Slip == 1]
#test_Stable = test_df[test_df.Stable == 1]

# Set X_train equal to 80% of the fraudulent transactions.
#X_train = Slip.sample
#count_Slips = len(Slip)

Slip = df[df.label == 0]
Stable = df[df.label == 1]
print ('Number of Stable class examples: ', Stable.label.count())
print ('Numbuer of Unstable class examples: ', Slip.label.count())


# In[27]:



# Add 80% of the normal transactions to X_train.
X_train = Slip.sample(frac=0.8)
count_Slip = len(X_train)

# Add 80% of the normal transactions to X_train.
X_train = pd.concat([X_train, Stable.sample(frac = 0.8)], axis = 0)

# X_test contains all the transaction not in X_train.
X_test = df.loc[~df.index.isin(X_train.index)]

# Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(X_test)

X_train = X_train.iloc[0:48600, :]
X_test = X_test.iloc[0:12150, :]



print ('Length of train set', len(X_train))
print ('Length of test set', len (X_test))



# In[28]:


# Add our target features to y_train and y_test.

#df.loc[df.label == 0, 'Slip'] = 1


y_train = X_train.label
y_test = X_test.label

# Drop target features from X_train and X_test.
X_train = X_train.drop(['label'], axis = 1)
X_test = X_test.drop(['label'], axis = 1)


# Check to ensure all of the training/testing dataframes are of the correct length



# In[29]:


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


# In[30]:


num_epochs = 10
total_series_length = len(X_train)
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 2
num_batches = total_series_length//batch_size//truncated_backprop_length
# Network Parameters
num_input = len(X_train)
num_hidden = int(len(X_train) * .5)
num_layers = 3


# In[ ]:



tf.reset_default_graph()

learning_rate = 0.001
training_epochs = 10
batch_size = 10
display_step = 20

#X_train = X_train.iloc[:1000,:]
# Network Parameters
num_input = len(X_train)
print (num_input)
timesteps = 1
num_hidden = int(X_train.iloc[:, 1].size * .5)
num_classes = 2 

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


#tf.reset_default_graph()


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    with tf.variable_scope('lstm1'):
        outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']


logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for epoch in range(1, training_epochs+1):
        print ("Running...")
        for batch in range(int(num_input/batch_size)):
            batch_x = X_train[batch*batch_size : (1+batch)*batch_size]
            batch_y = y_train[batch*batch_size : (1+batch)*batch_size]
            
            batch_x = batch_x.values.reshape((batch_size, -1))
            #print ("Batch shape", batch_x.shape)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            
            if epoch % display_step == 0 or epoch == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                #print("Step " + str(epoch) + ", Minibatch Loss= " + \
                 #     "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  #    "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for test data
    
    test_data = X_test.iloc[:1000, :].values.reshape((-1, timesteps, num_input))
    test_label = y_test[:test_len]
    print("Testing Accuracy:",         sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
    
    print (prediction.eval(feed_dict = {X: test_data, Y: test_label}))
    


