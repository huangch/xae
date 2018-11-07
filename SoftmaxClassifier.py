from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# import argparse
# 
# # Import data
# from tensorflow.examples.tutorials.mnist import input_data
# 
# import tensorflow as tf
# 
# FLAGS = None
# 
# def main(_):
#   mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
# 
#   # Create the model
#   x = tf.placeholder(tf.float32, [None, 784])
#   W = tf.Variable(tf.zeros([784, 10]))
#   b = tf.Variable(tf.zeros([10]))
#   y = tf.matmul(x, W) + b
# 
#   # Define loss and optimizer
#   y_ = tf.placeholder(tf.float32, [None, 10])
# 
#   # The raw formulation of cross-entropy,
#   #
#   #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
#   #                                 reduction_indices=[1]))
#   #
#   # can be numerically unstable.
#   #
#   # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
#   # outputs of 'y', and then average across the batch.
#   cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
#   train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# 
#   sess = tf.InteractiveSession()
#   # Train
#   tf.initialize_all_variables().run()
#   for _ in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# 
#   # Test trained model
#   correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#   print(sess.run(y,
#                  feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# #  print(sess.run(correct_prediction, feed_dict={x: mnist.test.images,
# #                                      y_: mnist.test.labels}))
#   print(sess.run(accuracy,
#                  feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# 
# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--data_dir', type=str, default='/tmp/data',
#                       help='Directory for storing data')
#   FLAGS = parser.parse_args()
#   tf.app.run()

def SoftmaxClassifier(
    input_dimensions = 784,
    output_dimensions = 10,
    initial_weight = None,
    initial_bias = None,
    ):
            
    keep_prob_input = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, input_dimensions])
    # W = tf.Variable(tf.zeros([input_dimensions, output_dimensions])) if initial_weight == None else tf.Variable(tf.constant(initial_weight))
    # b = tf.Variable(tf.zeros([output_dimensions])) if initial_bias == None else tf.Variable(tf.constant(initial_bias))     
    # y = tf.matmul(x, W) + b
    
    hidden_dimensions = 64
    
    W = tf.Variable(tf.zeros([input_dimensions, output_dimensions]))
    b = tf.Variable(tf.zeros([output_dimensions]))
    y_logits = tf.matmul(x, W) + b
    
    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, output_dimensions])   
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_logits, labels=y_))  
    
    y = tf.nn.softmax(y_logits)
    prediction = tf.argmax(y, 1)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    return {
        'cost': cost,
        'W': W,
        'b': b,
        'x': x,
        'y': y,
        'y_': y_,
        'prediction': prediction, 
        'accuracy': accuracy, 
        'keep_prob_input': keep_prob_input, 'keep_prob': keep_prob,
    } 
    