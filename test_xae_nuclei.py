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
import random
from tensorflow.examples.tutorials.mnist import input_data
from eXclusiveAutoencoder import eXclusiveAutoencoder
from SoftmaxClassifier import SoftmaxClassifier
from FullyConnectedNetwork import FullyConnectedNetwork
from scipy import io as spio
import cPickle as pickle
import csv
from sklearn.datasets import make_classification
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score, classification_report, confusion_matrix, roc_curve, auc
from datetime import datetime
import sys
import matplotlib
from scipy import interp
import matplotlib.pyplot as plt

class NucleiDataset():
    def __init__(self, mat_file, random_seed, k, m_list):
        data_dict = spio.loadmat(mat_file)
        
        train_x = data_dict['train_x']
        train_y = data_dict['train_y']
        
        sample_list_initial = range(train_x.shape[3])
        random.seed(random_seed)
        random.shuffle(sample_list_initial)
        sample_list_partitions = self.partition(sample_list_initial, k)
        
        sample_list = []
        for m in m_list:
            sample_list += sample_list_partitions[m]
        
        data = np.concatenate((
            train_x[::1, ::1, :, sample_list], 
            train_x[::1, ::-1, :, sample_list],
            train_x[::-1, ::1, :, sample_list], 
            train_x[::-1, ::-1, :, sample_list],
            ), axis=3).transpose((3,2,0,1))
        
        self.data = data.reshape((data.shape[0], data.shape[1]*data.shape[2]*data.shape[3]))
        
        label = np.concatenate((train_y[:, sample_list],train_y[:, sample_list],train_y[:, sample_list],train_y[:, sample_list]), axis=1)[0]
        
        self.label = np.zeros((label.shape[0], 2))
        
        for i in range(label.shape[0]):
            self.label[i][label[i]-1] = 1.0
            
    def __getitem__(self, index):
        return self.data[index], self.label[index] 
    
    def __len__(self):
        return self.label.shape[0]       
            
    def partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]                  
        
def test_nuclei():
    '''Test the convolutional autoencder using nuclei.'''
    # %%
    
    random.seed(datetime.now())
    RANDOM_SEED_MAGIC = random.randint(0,65536)
    
    nuclei_training_dataset = NucleiDataset(mat_file = 'tmi/training/training.mat', random_seed=RANDOM_SEED_MAGIC, k=10, m_list = [0,1,2,3, 4, 5, 6, 7, 8])
    nuclei_test_dataset = NucleiDataset(mat_file = 'tmi/training/training.mat', random_seed=RANDOM_SEED_MAGIC, k=10, m_list = [9])    
    nuclei_data, _ = nuclei_training_dataset[random.sample(range(nuclei_training_dataset.__len__()), 1000)]
    mean_img = np.mean(nuclei_data, axis=0)
    
    xae_learning_rate = 0.01
    smc_learning_rate = 0.0001
    n_xae_batch_size = 1000
    n_smc_batch_size = 1000
    n_xae_epochs = 10000
    n_smc_epochs = 20000
    n_xae_reload_per_epochs = 1000
    n_xae_display_per_epochs = 1000
    n_smc_reload_per_epochs = 1000
    n_smc_display_per_epochs = 1000
    
    xae_layers = [         
        {
            'n_channels': 144,
            'reconstructive_regularizer': 1.0, 
            'weight_decay': 1.0, 
            'sparse_regularizer': 1.0, 
            'sparsity_level': 0.05,
            'exclusive_regularizer': 1.0,
            'corrupt_prob': 1.0,
            'tied_weight': True,
            'exclusive_type': 'logcosh',
            'exclusive_scale': 10.0,    
            'gaussian_mean': 0.0,    
            'gaussian_std': 0.0,    
            'encode':'sigmoid', 'decode':'linear',
            'pathways': [
                range(0, 96),
                range(48, 144),
            ],
        },                                                                                           
    ]
     
    ae = eXclusiveAutoencoder(
        input_dimensions = nuclei_data.shape[1],
        layers = xae_layers,
        
        init_encoder_weight = None,
        init_decoder_weight = None,
        init_encoder_bias = None,
        init_decoder_bias = None,
    )
    
    smc = SoftmaxClassifier(
        input_dimensions = 144,
        output_dimensions = 2,
    )    
    
#     smc = FullyConnectedNetwork(
#         dimensions = [1024, 256, 128, 64, 2],
#         init_weight = None,
#         init_bias = None,  
#     )
    
    xae_optimizer_list = []  
     
    for layer_i in range(len(xae_layers)):
        xae_optimizer_list.append(tf.train.AdamOptimizer(xae_learning_rate).minimize(ae['layerwise_cost'][layer_i]['total'], var_list=[
                ae['encoder_weight'][layer_i],
                ae['encoder_bias'][layer_i],
                # ae['decoder_weight'][layer_i],
                # ae['decoder_bias'][layer_i],
        ]))
         
    xae_optimizer_full = tf.train.AdamOptimizer(xae_learning_rate).minimize(ae['cost']['total'])
    
    smc_optimizer = tf.train.AdamOptimizer(smc_learning_rate).minimize(smc['cost'])
    # smc_optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(smc['cost'])
    
    # fcn_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(fc['cost'])
    
    # correct_prediction = tf.equal(tf.argmax(smc['y'], 1), tf.argmax(smc['y_'], 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # %%
    # We create a session to use the graph
    sess = tf.Session()
    writer = tf.summary.FileWriter('logs', sess.graph)
    sess.run(tf.global_variables_initializer())
 
    # %%
    # Fit all training data
         
    for layer_i in range(len(xae_layers)):
        for epoch_i in range(n_xae_epochs): 
            if (epoch_i) % n_xae_reload_per_epochs == 0:
                batch_xs, batch_ys = nuclei_training_dataset[random.sample(range(nuclei_training_dataset.__len__()), n_xae_batch_size)]
                train = []
                train.append(np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                    batch_ys[:, 0], 
                    # batch_ys[:, 1],                    
                ]) == 1, axis=0))]]))
                train.append(np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                    # batch_ys[:, 0], 
                    batch_ys[:, 1],
                ]) == 1, axis=0))]]))
                     
            sess.run(xae_optimizer_list[layer_i], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                    
            if (epoch_i+1) % n_xae_display_per_epochs == 0:
                cost_total = sess.run(ae['layerwise_cost'][layer_i]['total'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                cost_reconstruction_error = sess.run(ae['layerwise_cost'][layer_i]['reconstruction_error'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                cost_sparsity = sess.run(ae['layerwise_cost'][layer_i]['sparsity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                cost_exclusivity = sess.run(ae['layerwise_cost'][layer_i]['exclusivity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                cost_weight_decay = sess.run(ae['layerwise_cost'][layer_i]['weight_decay'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                 
                print('layer:', layer_i+1, ', epoch:', epoch_i+1, ', total cost:', cost_total, ', recon error:', cost_reconstruction_error, ', sparsity:', cost_sparsity, ', weight decay:', cost_weight_decay, ', exclusivity: ', cost_exclusivity)
                       
                test_xs, test_ys = nuclei_training_dataset[random.sample(range(nuclei_training_dataset.__len__()), 2000)]
                
                test_xs_0 = np.array([img - mean_img for img in test_xs[np.where(np.any(np.array([
                    test_ys[:, 0], 
                    # test_ys[:, 1], 
                ]) == 1, axis=0))][:144]])
                recon_0 = sess.run(ae['layerwise_y'][layer_i], feed_dict={ae['x']: test_xs_0})

                test_xs_1 = np.array([img - mean_img for img in test_xs[np.where(np.any(np.array([
                    # test_ys[:, 0], 
                    test_ys[:, 1], 
                ]) == 1, axis=0))][:144]])
                recon_1 = sess.run(ae['layerwise_y'][layer_i], feed_dict={ae['x']: test_xs_1})
                                   
                weights = sess.run(ae['encoder_weight'][0])
                display_color_network(weights, filename='nuclei_weights.png')                             
                display_color_network(test_xs_0.transpose(), filename='nuclei_test_0.png')
                display_color_network(recon_0.transpose(), filename='nuclei_results_0.png')              
                display_color_network(test_xs_1.transpose(), filename='nuclei_test_1.png')
                display_color_network(recon_1.transpose(), filename='nuclei_results_1.png')              
 
    for epoch_i in range(n_xae_epochs):
        if (epoch_i) % n_xae_reload_per_epochs == 0:
            batch_xs, batch_ys = nuclei_training_dataset[random.sample(range(nuclei_training_dataset.__len__()), n_xae_batch_size)]
            train = []
            train.append(np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                batch_ys[:, 0], 
                # batch_ys[:, 1],
            ]) == 1, axis=0))]]))
            train.append(np.array([img - mean_img for img in batch_xs[np.where(np.any(np.array([
                # batch_ys[:, 0], 
                batch_ys[:, 1],
            ]) == 1, axis=0))]]))
        
        sess.run(xae_optimizer_full, feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
                      
        if (epoch_i+1) % n_xae_display_per_epochs == 0:
            cost_total = sess.run(ae['cost']['total'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_reconstruction_error = sess.run(ae['cost']['reconstruction_error'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_sparsity = sess.run(ae['cost']['sparsity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_exclusivity = sess.run(ae['cost']['exclusivity'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            cost_weight_decay = sess.run(ae['cost']['weight_decay'], feed_dict={ae['training_x'][0]: train[0], ae['training_x'][1]: train[1]})
            
            print('layer: full,', 'epoch:', epoch_i+1, ', total cost:', cost_total, ', recon error:', cost_reconstruction_error, ', sparsity:', cost_sparsity, ', weight decay:', cost_weight_decay, ', exclusivity: ', cost_exclusivity)
     
            n_examples = 5120
            test_xs, test_ys = nuclei_training_dataset[random.sample(range(nuclei_training_dataset.__len__()), n_examples)]
            
            test_xs_0 = np.array([img - mean_img for img in test_xs[np.where(np.any(np.array([
                test_ys[:, 0], 
                # test_ys[:, 1],
            ]) == 1, axis=0))][:144]])
            
            recon_0 = sess.run(ae['y'], feed_dict={ae['x']: test_xs_0})

            test_xs_1 = np.array([img - mean_img for img in test_xs[np.where(np.any(np.array([
                # test_ys[:, 0], 
                test_ys[:, 1],
            ]) == 1, axis=0))][:144]])
            
            recon_1 = sess.run(ae['y'], feed_dict={ae['x']: test_xs_1})
                         
            weights = sess.run(ae['encoder_weight'][0])
            display_color_network(weights, filename='nuclei_weights.png')                             
            display_color_network(test_xs_0.transpose(), filename='nuclei_test_0.png')
            display_color_network(recon_0.transpose(), filename='nuclei_results_0.png')
            display_color_network(test_xs_1.transpose(), filename='nuclei_test_1.png')
            display_color_network(recon_1.transpose(), filename='nuclei_results_1.png')
            
    for epoch_i in range(n_smc_epochs):
        if (epoch_i) % n_smc_reload_per_epochs == 0:
            batch_xs, batch_ys = nuclei_training_dataset[random.sample(range(nuclei_training_dataset.__len__()), n_smc_batch_size)]
              
        ae_z = sess.run(ae['z'], feed_dict={ae['x']: batch_xs})
        
        sess.run(smc_optimizer, feed_dict={smc['x']: ae_z, smc['y_']: batch_ys, smc['keep_prob_input']:0.9, smc['keep_prob']: 0.9  })
            
        if (epoch_i+1) % n_smc_display_per_epochs == 0:
            cost = sess.run(smc['cost'], feed_dict={smc['x']: ae_z, smc['y_']: batch_ys, smc['keep_prob_input']: 1.0, smc['keep_prob']: 1.0})
            
            test_xs, test_ys = nuclei_test_dataset[random.sample(range(nuclei_test_dataset.__len__()), n_smc_batch_size)]
            ae_z = sess.run(ae['z'], feed_dict={ae['x']: test_xs})
            
            acc = sess.run(smc['accuracy'], feed_dict={smc['x']: ae_z, smc['y_']: test_ys, smc['keep_prob_input']: 1.0, smc['keep_prob']: 1.0})
            
            print('epoch:', epoch_i+1, ', cost:', cost, ', accuracy:', acc)
                 
    test_xs, test_ys = nuclei_test_dataset[:]
    ae_z = sess.run(ae['z'], feed_dict={ae['x']: test_xs})
    
    y = sess.run(smc['y'], feed_dict={smc['x']: ae_z, smc['y_']: test_ys, smc['keep_prob_input']: 1.0, smc['keep_prob']: 1.0})
    prediction = sess.run(smc['prediction'], feed_dict={smc['x']: ae_z, smc['y_']: test_ys, smc['keep_prob_input']: 1.0, smc['keep_prob']: 1.0})
    conf_matrix = confusion_matrix(np.argmax(test_ys, axis=1), prediction)
    
#     with open('nuclei_results_0.csv', "wb") as csv_file:
#         csvwriter = csv.writer(csv_file, delimiter=str(u','))
#         for (p, y, g) in zip(prediction[:, 1], y[:, 1], test_ys[:, 1]):
#             csvwriter.writerow([p, y, g])
    
    num_labels = 2
    
    print('final f1_score:', f1_score(test_ys[:, 1], prediction, average="macro"))
    print('final precision:', precision_score(test_ys[:, 1], prediction, average="macro"))
    print('final recall:', recall_score(test_ys[:, 1], prediction, average="macro"))
    print('final average_precision:', average_precision_score(test_ys[:, 1], prediction, average="macro"))            
    print('final accuracy:', sess.run(smc['accuracy'], feed_dict={smc['x']: ae_z, smc['y_']: test_ys, smc['keep_prob_input']: 1.0, smc['keep_prob']: 1.0}))
            
                
    sys.stdout.write('\n\nConfusion Matrix')
    sys.stdout.write('\t'*(num_labels-2)+'| Accuracy')
    sys.stdout.write('\n'+'-'*8*(num_labels+1))
    sys.stdout.write('\n')
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[i])):
            sys.stdout.write(str(conf_matrix[i][j].astype(np.int))+'\t')
        sys.stdout.write('| %3.2f %%' % (conf_matrix[i][i]*100 / conf_matrix[i].sum()))
        sys.stdout.write('\n')
    sys.stdout.write('Number of test samples: %i \n\n' % conf_matrix.sum())
            
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_labels):
        fpr[i], tpr[i], _ = roc_curve(test_ys[:, i], y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_ys.ravel(), y.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_labels)]))
    
    # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(num_labels):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    # mean_tpr /= num_labels
    
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.3f})'
#                    ''.format(roc_auc["micro"]),
#              color='red', linestyle=':', linewidth=4)
#     
#     plt.plot(fpr["macro"], tpr["macro"],
#              label='macro-average ROC curve (area = {0:0.3f})'
#                    ''.format(roc_auc["macro"]),
#              color='blue', linestyle=':', linewidth=4)
    
    # cmap = matplotlib.cm.get_cmap('rainbow')
    # colors = [cmap(0.1+0.8*float(i)/float(num_labels-1)) for i in range(num_labels)]   
    # label_list = ['Epithelial', 'Inflammatory', 'Fibroblast', 'Miscellaneous']
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(num_labels), colors):
    plt.plot(
        fpr[0], 
        tpr[0], 
        color='darkorange', 
        lw=2, 
        label='ROC curve (area = {0:0.3f})'.format(roc_auc[0])
        )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.015])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")

    plt.gcf().savefig('rocplot.png')
    plt.clf()
                
    writer.close()
    
#     xae_encoder_weight = sess.run(ae['encoder_weight'][0])
#     xae_encoder_bias = sess.run(ae['encoder_bias'][0])
#     xae_decoder_weight = sess.run(ae['decoder_weight'][0])
#     xae_decoder_bias = sess.run(ae['decoder_bias'][0])
    
#     smc_weight = sess.run(smc['W'])
#     smc_bias = sess.run(smc['b'])
# 
#     data = {
#         'xae_encoder_weight': xae_encoder_weight,
#         'xae_encoder_bias': xae_encoder_bias,
#         'xae_decoder_weight': xae_decoder_weight,
#         'xae_decoder_bias': xae_decoder_bias,
#         'smc_weight': smc_weight,
#         'smc_bias': smc_bias,
#         }
#     
#     with open('nuclei.pickle', 'wb') as fp:
#         pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)    
 
if __name__ == '__main__':   
    test_nuclei()
