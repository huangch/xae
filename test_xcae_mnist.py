'''Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
'''
from __future__ import division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
from PIL import Image
import sys
from random import randint
import glob
from random import sample
from display_network import display_color_network
from display_network import display_network
from eXclusiveConvolutionalAutoencoder import eXclusiveConvolutionalAutoencoder
from time import time
from optparse import OptionParser
try:
   import cPickle as pickle
except:
   import pickle

# %%
def test_mnist():
    '''Test the convolutional autoencder using MNIST.'''
    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)

    learning_rate = 0.001
    batch_size = 1000
    n_epochs = 10000
    n_filter_size = 14
    n_reload_per_epochs = 100
    n_display_per_epochs = 1000
    input_shape = [None, 28, 28, 1]
    xae_layers =[
        {
            'n_channels': 144,
            'reconstructive_regularizer': 1.0, 
            'weight_decay': 1.0, 
            'sparse_regularizer': 1.0, 
            'sparsity_level': 0.05,
            'exclusive_regularizer': 1.0,
            'tied_weight': True,
            'conv_size': n_filter_size,
            'conv_stride': 2,
            'conv_padding': 'VALID',
            'pool_size': 8,
            'pool_stride': 1,
            'pool_padding': 'VALID',
            'corrupt_prob': 1.0,
            'encode':'lrelu', 'decode':'linear',
            'pathways': [
                range(0, 96),
                range(48, 144),
            ],
        },   
#         {
#             'n_channels': 256,
#             'reconstructive_regularizer': 1.0, 
#             'weight_decay': 1.0, 
#             'sparse_regularizer': 1.0, 
#             'sparsity_level': 0.05,
#             'exclusive_regularizer': 1.0,
#             'tied_weight': True,
#             'conv_size': 9,
#             'conv_stride': 1,
#             'conv_padding': 'VALID',
# #             'pool_size': 9,
# #             'pool_stride': 1,
# #             'pool_padding': 'VALID',
#             'corrupt_prob': 1.0,
#             'encode':'sigmoid', 'decode':'linear',
#             'pathways': [
#                 range(0, 96),
#                 range(48, 144),
#             ],
#         },                               
    ]
    
    ae = eXclusiveConvolutionalAutoencoder(
        input_shape = input_shape,
        layers = xae_layers,
        init_encoder_weight = None,
        init_decoder_weight = None,
        init_encoder_bias = None,
        init_decoder_bias = None,              
        )
 
    # %%
     
    optimizer_list = []  
      
    for layer_i in range(len(xae_layers)):
        optimizer_list.append(tf.train.AdamOptimizer(learning_rate).minimize(ae['layerwise_cost'][layer_i]['total'], var_list=[
                ae['encoder_weight'][layer_i],
                ae['encoder_bias'][layer_i],
                # ae['decoder_weight'][layer_i],
                # ae['decoder_bias'][layer_i],
        ]))
         
    optimizer_list.append(tf.train.AdamOptimizer(learning_rate).minimize(ae['cost']['total']))
     
    # %%
    # We create a session to use the graph
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
 
    # %%
    # Fit all training data
     
    for layer_i, (optimizer) in enumerate(optimizer_list):
        for epoch_i in range(n_epochs):
            if (epoch_i) % n_reload_per_epochs == 0:
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                train = []
                train.append(np.array([img.reshape((28, 28, 1)) - mean_img.reshape((28, 28, 1)) for img in batch_xs[np.where(np.any(np.array([
                    batch_ys[:, 0],
                    batch_ys[:, 1],
                    # batch_ys[:, 2],
                    # batch_ys[:, 3],
                    # batch_ys[:, 4], 
                    # batch_ys[:, 5],
                    # batch_ys[:, 6],
                    # batch_ys[:, 7],
                    # batch_ys[:, 8],
                    # batch_ys[:, 9],                    
                ]) == 1, axis=0))]]))
                train.append(np.array([img.reshape((28, 28, 1)) - mean_img.reshape((28, 28, 1)) for img in batch_xs[np.where(np.any(np.array([
                    # batch_ys[:, 0],
                    batch_ys[:, 1],
                    batch_ys[:, 2],
                    # batch_ys[:, 3],
                    # batch_ys[:, 4], 
                    # batch_ys[:, 5],
                    # batch_ys[:, 6],
                    # batch_ys[:, 7],
                    # batch_ys[:, 8],
                    # batch_ys[:, 9], 
                ]) == 1, axis=0))]]))            
            
            # sess.run(optimizer, feed_dict={ae['training_x'][0]: train[0], })
            sess.run(optimizer, feed_dict={
                ae['training_x'][0]: train[0],
                ae['training_x'][1]: train[1],
            })
                    
            if (epoch_i+1) % n_display_per_epochs == 0:
                data_dict = {
                    ae['training_x'][0]: train[0], 
                    ae['training_x'][1]: train[1],
                }
                
                if optimizer is optimizer_list[-1]:
                    cost_total = sess.run(ae['cost']['total'], feed_dict=data_dict)
                    cost_reconstruction_error = sess.run(ae['cost']['reconstruction_error'], feed_dict=data_dict)
                    cost_sparsity = sess.run(ae['cost']['sparsity'], feed_dict=data_dict)
                    cost_exclusivity = sess.run(ae['cost']['exclusivity'], feed_dict=data_dict)
                    cost_weight_decay = sess.run(ae['cost']['weight_decay'], feed_dict=data_dict)
                else:
                    cost_total = sess.run(ae['layerwise_cost'][layer_i]['total'], feed_dict=data_dict)
                    cost_reconstruction_error = sess.run(ae['layerwise_cost'][layer_i]['reconstruction_error'], feed_dict=data_dict)
                    cost_sparsity = sess.run(ae['layerwise_cost'][layer_i]['sparsity'], feed_dict=data_dict)
                    cost_exclusivity = sess.run(ae['layerwise_cost'][layer_i]['exclusivity'], feed_dict=data_dict)
                    cost_weight_decay = sess.run(ae['layerwise_cost'][layer_i]['weight_decay'], feed_dict=data_dict)
                
                print('layer:{}, epoch:{:5d}, cost:{:.6f}, error: {:.6f}, sparsity: {:.6f}, exclusivity: {:.6f}, weight decay: {:.6f}'.format(
                    optimizer is optimizer_list[-1] and 'A' or layer_i+1, 
                    epoch_i+1,
                    cost_total,
                    cost_reconstruction_error,
                    cost_sparsity,
                    cost_weight_decay,
                    cost_exclusivity))
                             
                n_examples = 5120
                test_xs, test_ys = mnist.test.next_batch(n_examples)  
                test_xs = np.array([img.reshape((28, 28, 1)) - mean_img.reshape((28, 28, 1)) for img in test_xs[np.where(np.any(np.array([
                    test_ys[:, 0], 
                    test_ys[:, 1], 
                    test_ys[:, 2],
                    # test_ys[:, 3], 
                    # test_ys[:, 4], 
                    # test_ys[:, 5],
                    # test_ys[:, 6], 
                    # test_ys[:, 7], 
                    # test_ys[:, 8],
                    # test_ys[:, 9],
                ]) == 1, axis=0))][:256]])
                
                if optimizer is optimizer_list[-1]:
                    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
                else:
                    recon = sess.run(ae['layerwise_y'][layer_i], feed_dict={ae['x']: test_xs})
                   
                weights = sess.run(ae['encoder_weight'][0])
                weights = weights.transpose((3,0,1,2)).reshape((144, n_filter_size*n_filter_size)).transpose()
                
                display_network(weights, filename='mnist_weight.png')
                display_network(test_xs.reshape((256,784)).transpose(), filename='mnist_test.png')
                display_network(recon.reshape((256,784)).transpose(), filename='mnist_results.png')                
 
 
def nn():
    # Parameters
    learning_rate = 0.1
    num_steps = 500
    batch_size = 128
    display_step = 100
    
    # Network Parameters
    n_hidden_1 = 256 # 1st layer number of neurons
    n_hidden_2 = 256 # 2nd layer number of neurons
    num_input = 784 # MNIST data input (img shape: 28*28)
    num_classes = 10 # MNIST total classes (0-9 digits)
    
    # tf Graph input
    X = tf.placeholder("float", [None, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }
    
    
    # Create model# Creat 
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    
    # Construct model# Const 
    logits = neural_net(X)
    
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
    
        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    
        print("Optimization Finished!")
    
        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: mnist.test.images,
                                          Y: mnist.test.labels}))
            

# %%
if __name__ == '__main__':
    test_mnist()
