# import eXclusiveAutoencoder.eXclusiveAutoencoder
from __future__ import division, print_function, unicode_literals
import csv
import scipy.io as sio
import numpy as np
from eXclusiveAutoencoder import eXclusiveAutoencoder 
from NoisyAndPooling import NoisyAndPooling
import tensorflow as tf
from numpy.random import randint
from time import time
import sys
from random import shuffle
from sklearn import cross_validation, grid_search
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
import random
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score, average_precision_score, classification_report, confusion_matrix
import misvm

# Python code to remove duplicate elements
         
class Musk1NormMatDataset():
    def __init__(self, mat_file='mil_data/musk1norm_matlab.mat', random_seed = 1234, bag_id_list=range(1, 93)):
        MUSK1_N_INSTANCE = 476
        MUSK1_N_BAG = 92
        MUSK1_DIM = 166
        
        datafile = sio.loadmat(mat_file)
        bag_ids = np.asarray(datafile['bag_ids'].transpose()).reshape((MUSK1_N_INSTANCE))
        labels = np.asarray(datafile['labels'].todense().transpose()).reshape((MUSK1_N_INSTANCE))
        features = np.asarray(datafile['features'].todense())  
        
        random.seed(random_seed)
        random.shuffle(bag_id_list)        
        
        self.bag_ids = np.empty((0,))
        self.features = np.empty((0, MUSK1_DIM))
        labels_buf = np.empty((0,))
        
        for id in bag_id_list:
            id_list = np.where(bag_ids == id)[0].tolist()
            self.bag_ids = np.concatenate((self.bag_ids, bag_ids[id_list]), axis=0)
            labels_buf = np.concatenate((labels_buf, labels[id_list]), axis=0)
            self.features = np.concatenate((self.features, features[id_list, :]), axis=0)
            
        self.labels = np.array([[1, 1] if labels_buf[i] == 1 else [1, 0] for i in range(labels_buf.shape[0])])
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.bag_ids[index]
         
    def __len__(self):
        return self.labels.shape[0]

    def duplicate_removal(self, duplicate_list):
        final_list = []
        for num in duplicate_list:
            if num not in final_list:
                final_list.append(num)
        return final_list
    
    def partition(self, lst, n):
        division = len(lst) / float(n)
        return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ] 
    
class Musk1Dataset():
    MUSK1_SIZE = 476
    MUSK1_DIM = 166
    MUSK1_N_BAGS = 92

    def __init__(self, mat_file='mil_data/musk1norm_matlab.mat', random_seed = 1234, bag_id_list=range(1, 93)):
        datafile = sio.loadmat(mat_file)
        bag_ids = np.asarray(datafile['bag_ids'].transpose()).reshape((self.MUSK1_SIZE))
        labels = np.asarray(datafile['labels'].todense().transpose()).reshape((self.MUSK1_SIZE))
        features = np.asarray(datafile['features'].todense())  
        
        random.seed(random_seed)
        random.shuffle(bag_id_list)        
                
        self.labels = [None]*len(bag_id_list)
        self.features = [None]*len(bag_id_list)
        
        for i, (id) in enumerate(bag_id_list):
            self.labels[i] = np.array([1.0, 1.0]) if labels[np.where(bag_ids == id)][0] == 1 else np.array([1.0, 0.0])
            self.features[i] = features[np.where(bag_ids == id)[0], :]
                    
    def __getitem__(self, index):
        if type(index) is list:
            return [self.features[i] for i in index], [self.labels[i] for i in index]
        elif type(index) is int:
            return self.features[index], self.labels[index]
         
    def __len__(self):
        return len(self.labels)
    
def test_musk1(training_index_list, testing_index_list):
    '''Test the convolutional autoencder using Musk1.'''
    # %%
    
    random.seed(datetime.now())
    RANDOM_SEED_MAGIC = random.randint(0,65536)
    
    # bag_id_list=range(1, 93)
    # testing_id = bag_id_list[testing_index]
    # del bag_id_list[testing_index]
    
    musk1_training_dataset = Musk1Dataset(mat_file='mil_data/musk1norm_matlab.mat', random_seed=RANDOM_SEED_MAGIC, bag_id_list=training_index_list)    
    musk1_testing_dataset = Musk1Dataset(mat_file='mil_data/musk1norm_matlab.mat', random_seed=RANDOM_SEED_MAGIC, bag_id_list=testing_index_list)
        
    musk1_data_list, _ = musk1_training_dataset[range(musk1_training_dataset.__len__())]
    musk1_data = np.concatenate(musk1_data_list)
    musk1_mean = np.mean(musk1_data, axis=0)
    
    xae_learning_rate = 0.001
    n_xae_batch_size = 32
    n_xae_epochs = 1000
    n_xae_reload_per_epochs = 100
    n_xae_display_per_epochs = 100
    
    nap_learning_rate = 0.0001
    n_nap_batch_size = 32
    n_nap_epochs = 10000
    n_nap_reload_per_epochs = 100
    n_nap_display_per_epochs = 10000
    
    xae_layers = [         
        {
            'n_channels': 128,
            'reconstructive_regularizer': 10.0, 
            'weight_decay': 1.0, 
            'sparse_regularizer': 1.0, 
            'sparsity_level': 0.05,
            'exclusive_regularizer': 1.0,
            'corrupt_prob': 1.0,
            'tied_weight': True,
            'encode':'sigmoid', 'decode':'linear',
            'pathways': [
                range(0, 64),
                range(0, 128),
            ],
        },
        {
            'n_channels': 64,
            'reconstructive_regularizer': 10.0, 
            'weight_decay': 1.0, 
            'sparse_regularizer': 1.0, 
            'sparsity_level': 0.05,
            'exclusive_regularizer': 1.0,
            'corrupt_prob': None,
            'tied_weight': True,
            'encode':'sigmoid', 'decode':'linear',
            'pathways': [
                range(0, 32),
                range(0, 64),
            ],
        },  
        {
            'n_channels': 32,
            'reconstructive_regularizer': 10.0, 
            'weight_decay': 1.0, 
            'sparse_regularizer': 1.0, 
            'sparsity_level': 0.05,
            'exclusive_regularizer': 1.0,
            'corrupt_prob': None,
            'tied_weight': True,
            'encode':'sigmoid', 'decode':'linear',
            'pathways': [
                range(0, 16),
                range(0, 32),
            ],
        },              
    ]
     
    xae = eXclusiveAutoencoder(
        input_dimensions = musk1_data.shape[1],
        layers = xae_layers,
        
        init_encoder_weight = None,
        init_decoder_weight = None,
        init_encoder_bias = None,
        init_decoder_bias = None,
        )
    
    nap = NoisyAndPooling(
        n_cls = 2,
        n_dim = 32,
        keep_prob = 0.75,
        pathways = [
            range(0, 16),
            range(16, 32),
            ],
        )

    xae_optimizer_list = []  
     
    for layer_i in range(len(xae_layers)):
        xae_optimizer_list.append(tf.train.AdamOptimizer(xae_learning_rate).minimize(xae['layerwise_cost'][layer_i]['total'], var_list=[
                xae['encoder_weight'][layer_i],
                xae['encoder_bias'][layer_i],
                # ae['decoder_weight'][layer_i],
                # ae['decoder_bias'][layer_i],
        ]))
         
    xae_optimizer_full = tf.train.AdamOptimizer(xae_learning_rate).minimize(xae['cost']['total'])
    # nap_optimizer = tf.train.AdamOptimizer(nap_learning_rate).minimize(nap['cost'])
    nap_optimizer = tf.train.RMSPropOptimizer(nap_learning_rate).minimize(nap['cost'])
    
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
                batch_xs, batch_ys = musk1_training_dataset[random.sample(range(musk1_training_dataset.__len__()), n_xae_batch_size)]
                
                train = []
                train.append(np.concatenate([batch_xs[i] - np.matmul(np.ones((batch_xs[i].shape[0],1)), musk1_mean.reshape((1, 166))) for i in range(len(batch_xs)) if np.all(batch_ys[i] == np.array([1, 0]))]))                
                train.append(np.concatenate([batch_xs[i] - np.matmul(np.ones((batch_xs[i].shape[0],1)), musk1_mean.reshape((1, 166))) for i in range(len(batch_xs)) if np.all(batch_ys[i] == np.array([1, 1]))]))
                
                # train = []
                # train.append(np.array([img - musk1_mean for img in batch_xs[np.where(np.all(batch_ys == [1, 0], axis=1))]]))
                # train.append(np.array([img - musk1_mean for img in batch_xs[np.where(np.all(batch_ys == [1, 1], axis=1))]]))
                     
            sess.run(xae_optimizer_list[layer_i], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                    
            if (epoch_i+1) % n_xae_display_per_epochs == 0:
                cost_total = sess.run(xae['layerwise_cost'][layer_i]['total'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                cost_reconstruction_error = sess.run(xae['layerwise_cost'][layer_i]['reconstruction_error'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                cost_sparsity = sess.run(xae['layerwise_cost'][layer_i]['sparsity'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                cost_exclusivity = sess.run(xae['layerwise_cost'][layer_i]['exclusivity'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                cost_weight_decay = sess.run(xae['layerwise_cost'][layer_i]['weight_decay'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                 
                print('layer:', layer_i+1, ', epoch:', epoch_i+1, ', total cost:', cost_total, ', recon error:', cost_reconstruction_error, ', sparsity:', cost_sparsity, ', weight decay:', cost_weight_decay, ', exclusivity: ', cost_exclusivity)            
 
    for epoch_i in range(n_xae_epochs):
        if (epoch_i) % n_xae_reload_per_epochs == 0:
            batch_xs, batch_ys = musk1_training_dataset[random.sample(range(musk1_training_dataset.__len__()), n_xae_batch_size)]
            
            train = []
            train.append(np.concatenate([batch_xs[i] - np.matmul(np.ones((batch_xs[i].shape[0],1)), musk1_mean.reshape((1, 166))) for i in range(len(batch_xs)) if np.all(batch_ys[i] == np.array([1, 0]))]))
            train.append(np.concatenate([batch_xs[i] - np.matmul(np.ones((batch_xs[i].shape[0],1)), musk1_mean.reshape((1, 166))) for i in range(len(batch_xs)) if np.all(batch_ys[i] == np.array([1, 1]))]))
            
            # train = []
            # train.append(np.array([img - musk1_mean for img in batch_xs[np.where(np.all(batch_ys == [1, 0], axis=1))]]))
            # train.append(np.array([img - musk1_mean for img in batch_xs[np.where(np.all(batch_ys == [1, 1], axis=1))]]))
                    
        sess.run(xae_optimizer_full, feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
                      
        if (epoch_i+1) % n_xae_display_per_epochs == 0:
            cost_total = sess.run(xae['cost']['total'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
            cost_reconstruction_error = sess.run(xae['cost']['reconstruction_error'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
            cost_sparsity = sess.run(xae['cost']['sparsity'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
            cost_exclusivity = sess.run(xae['cost']['exclusivity'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
            cost_weight_decay = sess.run(xae['cost']['weight_decay'], feed_dict={xae['training_x'][0]: train[0], xae['training_x'][1]: train[1]})
            
            print('layer: full,', 'epoch:', epoch_i+1, ', total cost:', cost_total, ', recon error:', cost_reconstruction_error, ', sparsity:', cost_sparsity, ', weight decay:', cost_weight_decay, ', exclusivity: ', cost_exclusivity)

    training_xs, training_ys = musk1_training_dataset[range(musk1_training_dataset.__len__())]
    train_bags = [sess.run(xae['z'], feed_dict={xae['x']: training_xs[i]-np.matmul(np.ones((training_xs[i].shape[0],1)), musk1_mean.reshape((1, 166)))}) for i in range(len(musk1_training_dataset))]
    train_labels = [-1*(2*training_ys[i][1]-1) for i in range(len(musk1_training_dataset))]
    
    testing_xs, testing_ys = musk1_testing_dataset[range(musk1_testing_dataset.__len__())]
    test_bags = [sess.run(xae['z'], feed_dict={xae['x']: testing_xs[i]-np.matmul(np.ones((testing_xs[i].shape[0],1)), musk1_mean.reshape((1, 166)))}) for i in range(len(musk1_testing_dataset))]
    test_labels = [-1*(2*testing_ys[i][1]-1) for i in range(len(musk1_testing_dataset))]
    
#     # Construct classifiers
#     classifiers = {}
#     classifiers['MissSVM'] = misvm.MissSVM(kernel='rbf', C=1.0, max_iters=10)
#     classifiers['sbMIL'] = misvm.sbMIL(kernel='linear', eta=0.1, C=1.0)
#     classifiers['SIL'] = misvm.SIL(kernel='linear', C=1.0)
# 
#     # Train/Evaluate classifiers
#     accuracies = {}
#     for algorithm, classifier in classifiers.items():
#         classifier.fit(train_bags, train_labels)
#         predictions = classifier.predict(test_bags)
#         accuracies[algorithm] = np.average(test_labels == np.sign(predictions))
# 
#     for algorithm, accuracy in accuracies.items():
#         print('\n%s Accuracy: %.1f%%' % (algorithm, 100 * accuracy))
# 
#     return accuracies['SIL']
    
    for epoch_i in range(n_nap_epochs):
        if (epoch_i) % n_nap_reload_per_epochs == 0:
            batch_xs, batch_ys, _ = musk1_training_dataset[random.sample(range(musk1_training_dataset.__len__()), n_nap_batch_size)]
 
            train = []
            train.append(np.concatenate([batch_xs[i] for i in range(len(batch_xs)) if np.all(batch_ys[i] == np.array([1, 0]))]))
            train.append(np.concatenate([batch_xs[i] for i in range(len(batch_xs)) if np.all(batch_ys[i] == np.array([1, 1]))]))

            # train = []
            # train.append(np.array([img - musk1_mean for img in batch_xs[np.where(np.all(batch_ys == [1, 0], axis=1))]]))
            # train.append(np.array([img - musk1_mean for img in batch_xs[np.where(np.all(batch_ys == [1, 1], axis=1))]]))

        ae_z = []
        ae_z.append(sess.run(xae['z'], feed_dict={xae['x']: train[0]}))
        ae_z.append(sess.run(xae['z'], feed_dict={xae['x']: train[1]}))
         
        sess.run(nap_optimizer, feed_dict={
            nap['training_x'][0]: ae_z[0], 
            nap['training_x'][1]: ae_z[1], 
            nap['training_t'][0]: np.array([1.0, 0.0]), 
            nap['training_t'][1]: np.array([1.0, 1.0]),
            nap['training_l'][0]: np.array([1.0, 0.0]), 
            nap['training_l'][1]: np.array([0.0, 1.0]),
            })
                     
        if (epoch_i+1) % n_nap_display_per_epochs == 0:
            cost = sess.run(nap['cost'], feed_dict={
                nap['training_x'][0]: ae_z[0], 
                nap['training_x'][1]: ae_z[1], 
                nap['training_t'][0]: np.array([1.0, 0.0]), 
                nap['training_t'][1]: np.array([1.0, 1.0]),
                nap['training_l'][0]: np.array([1.0, 0.0]), 
                nap['training_l'][1]: np.array([0.0, 1.0]),
                })            
             
            print('epoch:', epoch_i+1, ', cost:', cost[0])
 
    test_xs, test_ys, _ = musk1_testing_dataset[:]
     
    xae_z = sess.run(xae['z'], feed_dict={xae['x']: test_xs})
         
    prediction = sess.run(nap['prediction'], feed_dict={
        nap['x']: xae_z,
        })
     
    y = sess.run(nap['y'], feed_dict={
        nap['x']: xae_z,
        })
     
    print('prediction:', prediction[0], ', ground truth:', test_ys[0,1], ', y:', y[0])
     
    return prediction[0], test_ys[0,1], y[0,0], y[0,1] 

    writer.close()
    

def duplicate_removal(duplicate_list):
    final_list = []
    for num in duplicate_list:
        if num not in final_list:
            final_list.append(num)
    return final_list

def partition(lst, n):
    division = len(lst) / float(n)
    return [ lst[int(round(division * i)): int(round(division * (i + 1)))] for i in xrange(n) ]
    
if __name__ == '__main__':
    
    K_cross_validation = 10
    bag_id_list = range(1, 93)
    bag_id_list_list = partition(bag_id_list, K_cross_validation)
    
    accuracy = 0
    
    for i in range(K_cross_validation):
        training_id_list = []
        testing_id_list = bag_id_list_list[i]
        
        for j in range(K_cross_validation):
            if j != i:
                training_id_list += bag_id_list_list[j]
        
        accuracy += test_musk1(training_id_list, testing_id_list)
        
    accuracy /= K_cross_validation        
    print('\n%s Accuracy: %.1f%%' % ('total', 100 * accuracy))
    
#     with open('mil_musk1.csv', 'w') as csvfile:
#         fieldnames = ['id', 'prediction', 'ground_truth', 'y']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#          
#         result = np.empty((92, 3))
#         y_test = np.empty((92, 2))
#         y_score = np.empty((92, 2))
#         writer.writeheader()
#          
#         for i in range(92):
#             prediction, ground_truth, y0, y1 = test_musk1(i)        
#             writer.writerow({'id': i+1, 'prediction': prediction, 'ground_truth': ground_truth, 'y': y1})
#             result[i, 0] = ground_truth
#             result[i, 1] = y1
#             result[i, 2] = prediction
#              
#             y_test[i, :] = np.array([1, 0]) if ground_truth == 0 else np.array([0, 1])
#             y_score[i, :] = np.array([y0, y1])
#      
#         print('final f1_score:', f1_score(result[:, 0], result[:, 2], average="macro"))
#         print('final precision:', precision_score(result[:, 0], result[:, 2], average="macro"))
#         print('final recall:', recall_score(result[:, 0], result[:, 2], average="macro"))
#         print('final average_precision:', average_precision_score(result[:, 0], result[:, 2], average="macro"))            
#         print('final accuracy:', accuracy_score(result[:, 0], result[:, 2]))
# 
#         fig = plt.figure()
#         # Compute ROC curve and ROC area for each class
#         fpr, tpr, _ = roc_curve(y_test[:,1], y_score[:,1])
#         roc_auc = auc(fpr, tpr)
#         
#         # Compute micro-average ROC curve and ROC area
#         fpr_micro, tpr_micro, _ = roc_curve(y_test.ravel(), y_score.ravel())
#         roc_auc_micro = auc(fpr, tpr)
#         
#         # Plot of a ROC curve for a specific class  
#         plt.plot(fpr, tpr, color='darkorange',
#                 lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
#         plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#         plt.xlim([-0.01, 1.0])
#         plt.ylim([0.0, 1.01])
#         plt.xlabel('False Positive Rate')
#         plt.ylabel('True Positive Rate')
#         plt.title('Receiver Operating Characteristic (ROC) Curve')
#         plt.legend(loc="lower right")    
#         
#         plt.gcf().savefig('mil_musk1.png')   
