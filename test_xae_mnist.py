"""Tutorial on how to create an autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
"""
# %% Imports
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import scipy.io
from display_network import display_color_network
from display_network import display_network
import glob
from PIL import Image
from random import randint
from random import sample
# import stl10_input
# import tensorflow.examples.tutorials.mnist.input_data as input_data
from tensorflow.examples.tutorials.mnist import input_data
from eXclusiveAutoencoder import eXclusiveAutoencoder
from AMSGrad import AMSGrad    
from sklearn.metrics import f1_score
 
def test_mnist():
    '''Test the convolutional autoencder using MNIST.'''
    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
 
    ae = eXclusiveAutoencoder(
        input_dimensions = 784,
        layers = [
            {
                'n_channels': 144,
                'reconstructive_regularizer': 1.0, 
                'weight_decay': 1.0, 
                'sparse_regularizer': 1.0, 
                'sparsity_level': 0.05,
                'exclusive_regularizer': 1.0,
                'exclusive_type': 'logcosh',
                'exclusive_logcosh_scale': 10.0,
                'corrupt_prob': 1.0,
                'tied_weight': True,
                'encode':'sigmoid', 'decode':'linear',
                'pathways': [
                    # range(0, 144),
                    range(0, 96),
                    range(48, 144),
                ],
            },                                                                                                 
        ],
        
        init_encoder_weight = None,
        init_decoder_weight = None,
        init_encoder_bias = None,
        init_decoder_bias = None,
    )
 
    # %%
    learning_rate = 0.01
    n_reload_per_epochs = 10
    n_display_per_epochs = 10000
    batch_size = 2000
    n_epochs = 100000
     
     
    optimizer_list = []  
     
    for layer_i in range(1):
        optimizer_list.append(AMSGrad(learning_rate).minimize(ae['layerwise_cost'][layer_i]['total'], var_list=[
                ae['encoder_weight'][layer_i],
                ae['encoder_bias'][layer_i],
                # ae['decoder_weights'][layer_i],
                # ae['decoder_biases'][layer_i],
        ]))
         
    # optimizer_full = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost']['total'])
    
    optimizer_list.append(AMSGrad(learning_rate).minimize(ae['cost']['total']))
     
    # %%
    # We create a session to use the graph
    sess = tf.Session()
    writer = tf.summary.FileWriter('logs', sess.graph)
    sess.run(tf.global_variables_initializer())
 
    # %%
    # Fit all training data
         
    for optimizer_i, (optimizer) in enumerate(optimizer_list):
        for epoch_i in range(n_epochs): 
            if (epoch_i) % n_reload_per_epochs == 0:
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                
                
                batch_x0 = np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                    batch_ys[:, 0], 
                    # batch_ys[:, 1],
                    # batch_ys[:, 2], 
                    # batch_ys[:, 3],
                    # batch_ys[:, 4], 
                    # batch_ys[:, 5],
                    # batch_ys[:, 6], 
                    # batch_ys[:, 7],
                    # batch_ys[:, 8], 
                    # batch_ys[:, 9],
                ]) == 1, axis=0))]])
                
                batch_x1 = np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                    # batch_ys[:, 0], 
                    batch_ys[:, 1],
                    # batch_ys[:, 2], 
                    # batch_ys[:, 3],
                    # batch_ys[:, 4], 
                    # batch_ys[:, 5],
                    # batch_ys[:, 6], 
                    # batch_ys[:, 7],
                    # batch_ys[:, 8], 
                    # batch_ys[:, 9],
                ]) == 1, axis=0))]])
                
                batch_x2 = np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                    # batch_ys[:, 0], 
                    # batch_ys[:, 1],
                    batch_ys[:, 2], 
                    # batch_ys[:, 3],
                    # batch_ys[:, 4], 
                    # batch_ys[:, 5],
                    # batch_ys[:, 6], 
                    # batch_ys[:, 7],
                    # batch_ys[:, 8], 
                    # batch_ys[:, 9],
                ]) == 1, axis=0))]])                
                
                min_batch_size_x01 = np.min((batch_x0.shape[0], batch_x1.shape[0]))                
                batch_x01 = 0.5*(batch_x0[:min_batch_size_x01]+batch_x1[:min_batch_size_x01])
                
                min_batch_size_x012 = np.min((batch_x0.shape[0], batch_x1.shape[0], batch_x2.shape[0]))                
                batch_x012 = 0.333*(batch_x0[:min_batch_size_x012]+batch_x1[:min_batch_size_x012]+batch_x2[:min_batch_size_x012])
                 
                batch_x1 = np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                    # batch_ys[:, 0], 
                    batch_ys[:, 1],
                    # batch_ys[:, 2], 
                    # batch_ys[:, 3],
                    # batch_ys[:, 4], 
                    # batch_ys[:, 5],
                    # batch_ys[:, 6], 
                    # batch_ys[:, 7],
                    # batch_ys[:, 8], 
                    # batch_ys[:, 9],
                ]) == 1, axis=0))]])
                 
                batch_x2 = np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                    # batch_ys[:, 0], 
                    # batch_ys[:, 1],
                    batch_ys[:, 2], 
                    # batch_ys[:, 3],
                    # batch_ys[:, 4], 
                    # batch_ys[:, 5],
                    # batch_ys[:, 6], 
                    # batch_ys[:, 7],
                    # batch_ys[:, 8], 
                    # batch_ys[:, 9],
                ]) == 1, axis=0))]])
                 
                min_batch_size_x12 = np.min((batch_x1.shape[0], batch_x2.shape[0]))
                 
                batch_x12 = 0.5*(batch_x1[:min_batch_size_x12]+batch_x2[:min_batch_size_x12])
# 




                
                train = []
                # train.append(batch_x012)
                train.append(batch_x01)
                train.append(batch_x12)
                
#                 train.append(np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
#                     batch_ys[:, 0], 
#                     batch_ys[:, 1],
#                     # batch_ys[:, 2], 
#                     # batch_ys[:, 3],
#                     # batch_ys[:, 4], 
#                     # batch_ys[:, 5],
#                     # batch_ys[:, 6], 
#                     # batch_ys[:, 7],
#                     # batch_ys[:, 8], 
#                     # batch_ys[:, 9],
#                 ]) == 1, axis=0))]]))
#                 train.append(np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
#                     # batch_ys[:, 0], 
#                     batch_ys[:, 1],
#                     batch_ys[:, 2], 
#                     # batch_ys[:, 3],
#                     # batch_ys[:, 4], 
#                     # batch_ys[:, 5],
#                     # batch_ys[:, 6], 
#                     # batch_ys[:, 7],
#                     # batch_ys[:, 8], 
#                     # batch_ys[:, 9],
#                 ]) == 1, axis=0))]]))
                     
            sess.run(optimizer, feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    
            if (epoch_i+1) % n_display_per_epochs == 0:
                if not optimizer is optimizer_list[-1]:
                    cost_total = sess.run(ae['layerwise_cost'][optimizer_i]['total'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    cost_reconstruction_error = sess.run(ae['layerwise_cost'][optimizer_i]['reconstruction_error'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    cost_sparsity = sess.run(ae['layerwise_cost'][optimizer_i]['sparsity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    cost_exclusivity = sess.run(ae['layerwise_cost'][optimizer_i]['exclusivity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    cost_weight_decay = sess.run(ae['layerwise_cost'][optimizer_i]['weight_decay'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                     
                    print('layer:', optimizer_i+1, ', epoch:', epoch_i+1, ', total cost:', cost_total, ', recon error:', cost_reconstruction_error, ', sparsity:', cost_sparsity, ', weight decay:', cost_weight_decay, ', exclusivity: ', cost_exclusivity)
                           
                else:
                    cost_total = sess.run(ae['cost']['total'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    cost_reconstruction_error = sess.run(ae['cost']['reconstruction_error'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    cost_sparsity = sess.run(ae['cost']['sparsity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    cost_exclusivity = sess.run(ae['cost']['exclusivity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    cost_weight_decay = sess.run(ae['cost']['weight_decay'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    
                    print('layer: full,', 'epoch:', epoch_i+1, ', total cost:', cost_total, ', recon error:', cost_reconstruction_error, ', sparsity:', cost_sparsity, ', weight decay:', cost_weight_decay, ', exclusivity: ', cost_exclusivity)

                           
                n_examples = 5120
                test_xs, test_ys = mnist.test.next_batch(n_examples)  
                
                test_xs = np.array([img - mean_img for img in test_xs[np.where(np.any(np.array([
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
                ]) == 1, axis=0))][:144]])
                
                if not optimizer is optimizer_list[-1]:
                    recon = sess.run(ae['layerwise_y'][layer_i], feed_dict={ae['x']: test_xs})
                else:
                    recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
                   
                weights = sess.run(ae['encoder_weight'][0])
                # weights = np.transpose(weights, axes=(3,0,1,2))
                
                # display_network(batch_x012[:144].transpose(), filename='mnist_batch_01.png')
                display_network(batch_x01[:144].transpose(), filename='mnist_batch_01.png')
                display_network(batch_x12[:144].transpose(), filename='mnist_batch_12.png')
                display_network(test_xs.transpose(), filename='mnist_test.png')
                display_network(recon.reshape((144,784)).transpose(), filename='mnist_results.png')
                display_network(weights, filename='mnist_weights.png')                             

                  
    writer.close()
    
    return ae
 

 
def nn(xae_model):
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    
    # Parameters
    learning_rate = 0.01
    num_steps = 10000
    batch_size = 1280
    test_size = 1000
    display_step = 1000
    
    # Network Parameters
    n_hidden_1 = 64 # 1st layer number of neurons
    n_hidden_2 = 64 # 2nd layer number of neurons
    num_input = 144 # MNIST data input (img shape: 28*28)
    num_classes = 3 # MNIST total classes (0-9 digits)
    
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
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = AMSGrad(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    
    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    prediction = tf.argmax(logits, 1)
    
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    
    # Start training
    with tf.Session() as sess:
    
        # Run the initializer
        sess.run(init)
    
        for step in range(1, num_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            
            batch_x = np.array([img - mean_img for img in batch_x[np.where(np.any(np.array([
                batch_y[:, 0], 
                batch_y[:, 1],
                batch_y[:, 2], 
                # batch_ys[:, 3],
                # batch_ys[:, 4], 
                # batch_ys[:, 5],
                # batch_ys[:, 6], 
                # batch_ys[:, 7],
                # batch_ys[:, 8], 
                # batch_ys[:, 9],
            ]) == 1, axis=0))]])
 
            batch_x = sess.run(xae_model['z'], feed_dict={xae_model['x']: batch_x})
            
            batch_y = np.array([label[0:3] for label in batch_y[np.where(np.any(np.array([
                batch_y[:, 0], 
                batch_y[:, 1],
                batch_y[:, 2], 
                # batch_ys[:, 3],
                # batch_ys[:, 4], 
                # batch_ys[:, 5],
                # batch_ys[:, 6], 
                # batch_ys[:, 7],
                # batch_ys[:, 8], 
                # batch_ys[:, 9],
            ]) == 1, axis=0))]])
            
            
            
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x[0:128], Y: batch_y[0:128]})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    
        print("Optimization Finished!")
    

        



        test_x, test_y = mnist.train.next_batch(test_size)
            
        test_x = np.array([img - mean_img for img in test_x[np.where(np.any(np.array([
            test_y[:, 0], 
            test_y[:, 1],
            test_y[:, 2], 
            # batch_ys[:, 3],
            # batch_ys[:, 4], 
            # batch_ys[:, 5],
            # batch_ys[:, 6], 
            # batch_ys[:, 7],
            # batch_ys[:, 8], 
            # batch_ys[:, 9],
        ]) == 1, axis=0))]])
        
        test_x = sess.run(xae_model['z'], feed_dict={xae_model['x']: test_x})
        
        test_y = np.array([label[0:3] for label in test_y[np.where(np.any(np.array([
            test_y[:, 0], 
            test_y[:, 1],
            test_y[:, 2], 
            # batch_ys[:, 3],
            # batch_ys[:, 4], 
            # batch_ys[:, 5],
            # batch_ys[:, 6], 
            # batch_ys[:, 7],
            # batch_ys[:, 8], 
            # batch_ys[:, 9],
        ]) == 1, axis=0))]])
           
        # Calculate accuracy for MNIST test images
        print("Testing Accuracy:", \
            sess.run(accuracy, feed_dict={X: test_x,
                                          Y: test_y}))
        
        pred_y = sess.run(prediction, feed_dict={X: test_x})
        true_y = np.argmax(test_y, axis=1)
        
        score = f1_score(true_y, pred_y, average='weighted')
        print("f1 score:", \
            score)
        
    
    return score    
            
if __name__ == '__main__':
    cnt = 5
    score_list = []
    for _ in range(cnt):
        xae_model = test_mnist()
        score_list.append(nn(xae_model))
        
    print('scores: ', score_list, ', mean: ', np.array(score_list).mean())
    
     
