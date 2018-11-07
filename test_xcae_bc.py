'''Tutorial on how to create a convolutional autoencoder w/ Tensorflow.

Parag K. Mital, Jan 2016
'''
from __future__ import division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import glob
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
import eXclusiveConvolutionalAutoencoder.xcae 
from time import time
from optparse import OptionParser
import math
import cPickle as pickle

def EstUsingMacenko( I, Io, beta, alpha ):
        
    if len(I.shape) == 2 and I.shape[1] == 3:
        vec = I
    else:  
        vec = np.array(I.reshape(I.shape[0]*I.shape[1], I.shape[2]),dtype=float)        
    
    OD = -np.log((vec+1.0)/Io)
    
    ODhat = OD[np.where([not any(t<beta) for t in OD])]
    C = np.cov(ODhat.transpose())
    
    if np.any(np.isnan(C)) or np.any(np.isnan(C)):
        raise ValueError('Infeasible covariance.')
            
    _, V = np.linalg.eig(C)
    # id = np.array([i for i in reversed(d)])
    iV = -1.0*np.array([[i for i in reversed(j)] for j in V])
    THETA = np.dot(ODhat, iV[:,1:3])
    PHI = np.array([math.atan2(t[1],t[0]) for t in THETA])
    minPhi = np.percentile(PHI, alpha)
    maxPhi = np.percentile(PHI, 100-alpha)
    
    VEC1 = np.dot(iV[:,1:3],np.array([[math.cos(minPhi)], [math.sin(minPhi)]]))
    VEC2 = np.dot(iV[:,1:3],np.array([[math.cos(maxPhi)], [math.sin(maxPhi)]]))
    
    if VEC1[0] > VEC2[0]:
        M = np.transpose(np.hstack((VEC1, VEC2)))
    else:
        M = np.transpose(np.hstack((VEC2, VEC1)))

    return M

def Deconvolve(I, Io, M):
    if M.shape[0] < 3:
        M = np.vstack((M, np.cross(M[0], M[1])))
    
    M = M/np.tile(np.reshape(np.sqrt(np.sum(np.power(M,2),axis=1)), (3,1)),[1,3])
    
    if len(I.shape) == 2 and I.shape[1] == 3:
        vec = np.array(I,dtype=float)
    else:
        vec = np.array(I.reshape(I.shape[0]*I.shape[1], I.shape[2]),dtype=float)
    
    Y = -np.log((vec+1.0)/Io)
    
    C = np.dot(Y,np.linalg.inv(M))
    
    if len(I.shape) == 2 and I.shape[1] == 3:
        DCh = C
    else:
        DCh = C.reshape((I.shape[0],I.shape[1],3))
    
    return DCh, M 

def NormMacenko(Source, Target, Io, beta, alpha):
#     if not 'Source' in kwargs:
#         raise ValueError('Parameter Source is needed.')
#     else:
#         Source = kwargs['Source']
#         
#     if not 'Target' in kwargs:
#         raise ValueError('Patameter Target is needed.')
#     else:
#         Target = kwargs['Target']
#         
#     if not 'Io' in kwargs:
#         Io = 255
#     else:
#         Io = kwargs['Io']
#         
#     if not 'beta' in kwargs:
#         beta = 0.15
#     else:
#         beta = kwargs['beta']
# 
#     if not 'alpha' in kwargs:
#         alpha = 1
#     else:
#         alpha = kwargs['alpha']
        
    MTarget = EstUsingMacenko(I=Target, Io=Io, beta=beta, alpha=alpha)
    C, MTarget = Deconvolve(I=Target, Io=Io, M=MTarget )
    
    if len(Target.shape) == 2 and Target.shape[1] == 3:
        C = np.array(C,dtype=float)
    else:
        C = np.array(C.reshape(Target.shape[0]*Target.shape[1], Target.shape[2] ),dtype=float)
    
    maxCTarget = np.percentile(C, alpha,axis=0)

    MSource = EstUsingMacenko(Source, Io, beta, alpha)
    C, MSource = Deconvolve(I=Source, Io=Io, M=MSource)
    
    if len(Source.shape) == 2 and Source.shape[1] == 3:
        C = np.array(C, dtype=float)
    else:
        C = np.array(C.reshape(Source.shape[0]*Source.shape[1], Source.shape[2]),dtype=float)
            
    maxCSource = np.percentile(C, alpha,axis=0)
    
    C = C/maxCSource
    C = C*maxCTarget
    
    Norm = Io*np.exp(np.dot(C, -MTarget))
    Norm[Norm>Io]=Io
    
    if len(Source.shape) == 2 and Source.shape[1] == 3:
        Norm = np.array(Norm,dtype=np.uint8)
    else:
        Norm = np.array(Norm.reshape((Source.shape[0],Source.shape[1],3)),dtype=np.uint8)
    
    return Norm

def read_data_sets(patch_size, sample_per_img):
    tif_list = glob.glob("BreastCancerCells/Breast Cancer Cells/*.tif")
    
    dataset = np.empty((len(tif_list)*sample_per_img, patch_size, patch_size, 3))
    labelset = np.zeros((len(tif_list)*sample_per_img, 2))
    
    idx = 0
    for file in tif_list:
        img = Image.open(file)
        img = np.array(img)
        
        xy_list = [xy for xy in zip(
            np.random.randint(0, img.shape[1]-patch_size, sample_per_img),
            np.random.randint(0, img.shape[0]-patch_size, sample_per_img),
        )]
        
        for xy in xy_list:
            patch = (1.0/255.0)*img[xy[1]:xy[1]+patch_size, xy[0]:xy[0]+patch_size].astype(np.float32)
            # patch = patch.reshape((patch_size, patch_size, 1))
            dataset[idx, ...] = patch 
            if 'malignant' in file:
                labelset[idx, 1] = 1.0 
            else:
                labelset[idx, 0] = 1.0
                
            idx += 1
            
    return dataset, labelset

def read_hematoxylin_data_sets(patch_size, sample_per_img):
    tif_list = glob.glob("BreastCancerCells/Breast Cancer Cells/hematoxylin/*.tif")
    
    dataset = np.empty((len(tif_list)*sample_per_img, patch_size * patch_size))
    labelset = np.empty((len(tif_list)*sample_per_img, 2))
        
    idx = 0
    for file in tif_list:
        img = np.array(Image.open(file))
        
        xy_list = [xy for xy in zip(
            np.random.randint(0, img.shape[1]-patch_size, sample_per_img),
            np.random.randint(0, img.shape[0]-patch_size, sample_per_img),
        )]
        
        for xy in xy_list:
            patch = img[xy[1]:xy[1]+patch_size, xy[0]:xy[0]+patch_size].astype(np.float32)
            dataset[idx, :] = patch.reshape((patch_size*patch_size))
            if 'malignant' in file:
                labelset[idx, :] = np.array([0.0, 1.0])
            else:
                labelset[idx, :] = np.array([1.0, 0.0])
                
            idx += 1
            
    return dataset, labelset

# %%
def test_bc():
    '''Test the convolutional autoencder using MNIST.'''
    # %%
    # load MNIST as before
    
    n_channels = 144
    filter_size = 28
    stride_size = 7
    input_size = 128
    n_epochs = 10000
    n_display = 100
    n_reload = 100
    patch_per_image = 10
    patch_per_display = 16
    
    test_xs, _ = read_hematoxylin_data_sets(input_size, patch_per_image)
    mean_img = np.mean(test_xs, axis=0)
    # mean_img = np.zeros((128, 128))
    
    ae = eXclusiveConvolutionalAutoencoder(
        input_shape = [None, input_size, input_size, 1],
         
        layers = [
            {
                'n_channels': n_channels,
                'reconstructive_regularizer': 1.0, 
                'weight_decay': 1.0, 
                'sparse_regularizer': 1.0, 
                'sparsity_level': 0.05,
                'exclusive_regularizer': 1.0,
                'tied_weight': True,
                'filter_size': filter_size,
                'stride_size': stride_size,
                'corrupt_prob': 0.5,
                'padding_type': 'SAME',
                'encode':'sigmoid', 'decode':'linear',
                'pathways': [
                    range(0, 128),
                    range(0, 256),
                ],
            },             
        ],
        
        init_encoder_weights = None,
        init_decoder_weights = None,
        init_encoder_biases = None,
        init_decoder_biases = None,              
            )
 
    # %%
    learning_rate = 0.01
     
    optimizer_list = []  
      
    for layer_i in range(1):
        optimizer_list.append(tf.train.AdamOptimizer(learning_rate).minimize(ae['layerwise_cost'][layer_i]['total'], var_list=[
                ae['encoder_weights'][layer_i],
                ae['encoder_biases'][layer_i],
                # ae['decoder_weights'][layer_i],
                # ae['decoder_biases'][layer_i],
        ]))
         
    optimizer_full = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost']['total'])
     
    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
 
    # %%
    # Fit all training data
    
    
     
    for layer_i in range(1):
        for epoch_i in range(n_epochs):
            if (epoch_i) % n_reload == 0:
                batch_xs, batch_ys = read_hematoxylin_data_sets(input_size, patch_per_image)
                train = []
                train.append(np.array([img.reshape((input_size, input_size, 1)) - mean_img.reshape((input_size, input_size, 1)) for img in batch_xs[np.where(np.any(np.array([
                    batch_ys[:, 0],
                    # batch_ys[:, 1],            
                ]) == 1, axis=0))]]))
                train.append(np.array([img.reshape((input_size, input_size, 1)) - mean_img.reshape((input_size, input_size, 1)) for img in batch_xs[np.where(np.any(np.array([
                    # batch_ys[:, 0],
                    batch_ys[:, 1],
                ]) == 1, axis=0))]]))            
            
            sess.run(optimizer_list[layer_i], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    
            if (epoch_i+1) % n_display == 0:
                cost_total = sess.run(ae['layerwise_cost'][layer_i]['total'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                cost_reconstruction_error = sess.run(ae['layerwise_cost'][layer_i]['reconstruction_error'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                cost_sparsity = sess.run(ae['layerwise_cost'][layer_i]['sparsity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                cost_exclusivity = sess.run(ae['layerwise_cost'][layer_i]['exclusivity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                cost_weight_decay = sess.run(ae['layerwise_cost'][layer_i]['weight_decay'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                 
                print('layer:{}, epoch:{:d}, cost:{:f}, recon:{:f}, sparsity:{:f}, weight:{:f}, exclusivity:{:f}'.format(
                    layer_i+1, 
                    epoch_i+1,
                    cost_total,
                    cost_reconstruction_error,
                    cost_sparsity,
                    cost_weight_decay,
                    cost_exclusivity))
                             
                test_xs, test_ys = read_hematoxylin_data_sets(input_size, patch_per_image)

                test_xs = np.array([img.reshape((input_size, input_size, 1)) - mean_img.reshape((input_size, input_size, 1)) for img in test_xs[np.where(np.any(np.array([
                    test_ys[:, 0], 
                    test_ys[:, 1], 
                ]) == 1, axis=0))][np.random.randint(0, test_xs.shape[0], patch_per_display)]])
                recon = sess.run(ae['layerwise_y'][layer_i], feed_dict={ae['x']: test_xs})
                   
                weights = sess.run(ae['encoder_weights'][0])
                bias = sess.run(ae['encoder_biases'][0])
                
                data = {'weights': weights, 'bias': bias}
                with open('bc.pickle', 'wb') as fp:
                    pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
                
                display_network(weights.transpose((3,0,1,2)).reshape((n_channels, filter_size*filter_size)).transpose(), filename='bc_weights.png')
                display_network(test_xs.reshape((patch_per_display,input_size*input_size)).transpose(), filename='bc_test.png')
                display_network(recon.reshape((patch_per_display,input_size*input_size)).transpose(), filename='bc_results.png')                
 
    for epoch_i in range(n_epochs):
        if (epoch_i) % n_reload == 0:
            batch_xs, batch_ys = read_hematoxylin_data_sets(input_size, patch_per_image)
            train = []
            train.append(np.array([img.reshape((input_size, input_size, 1)) - mean_img.reshape((input_size, input_size, 1)) for img in batch_xs[np.where(np.any(np.array([
                batch_ys[:, 0],
                # batch_ys[:, 1],          
            ]) == 1, axis=0))]]))
            train.append(np.array([img.reshape((input_size, input_size, 1)) - mean_img.reshape((input_size, input_size, 1)) for img in batch_xs[np.where(np.any(np.array([
                # batch_ys[:, 0],
                batch_ys[:, 1],
            ]) == 1, axis=0))]])) 
                        
        sess.run(optimizer_full, feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
              
        if (epoch_i+1) % n_display == 0:
            cost_total = sess.run(ae['cost']['total'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_reconstruction_error = sess.run(ae['cost']['reconstruction_error'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_sparsity = sess.run(ae['cost']['sparsity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_exclusivity = sess.run(ae['cost']['exclusivity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_exclusivity = sess.run(ae['cost']['exclusivity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_weight_decay = sess.run(ae['cost']['weight_decay'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            
            print('layer:{}, epoch:{:d}, cost:{:f}, recon:{:f}, sparsity:{:f}, weight:{:f}, exclusivity:{:f}'.format(
                'F', 
                epoch_i+1,
                cost_total,
                cost_reconstruction_error,
                cost_sparsity,
                cost_weight_decay,
                cost_exclusivity))

            test_xs, test_ys = read_hematoxylin_data_sets(input_size, patch_per_image) 
            test_xs = np.array([img.reshape((input_size, input_size, 1)) - mean_img.reshape((input_size, input_size, 1)) for img in test_xs[np.where(np.any(np.array([
                    test_ys[:, 0], 
                    test_ys[:, 1], 
            ]) == 1, axis=0))][np.random.randint(0, test_xs.shape[0], patch_per_display)]])
            recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
                             
            weights = sess.run(ae['encoder_weights'][0])
            bias = sess.run(ae['encoder_biases'][0])
            
            data = {'weights': weights, 'bias': bias}
            with open('bc.pickle', 'wb') as fp:
                pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
            display_network(weights.transpose((3,0,1,2)).reshape((n_channels, filter_size*filter_size)).transpose(), filename='bc_weights.png')          
            display_network(test_xs.reshape((patch_per_display,input_size*input_size)).transpose(), filename='bc_test.png')
            display_network(recon.reshape((patch_per_display,input_size*input_size)).transpose(), filename='bc_results.png')   
 
# %%
if __name__ == '__main__':
    test_bc()
    
