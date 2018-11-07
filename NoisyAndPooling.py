import tensorflow as tf
import math 

def NoisyAndPooling(
    n_cls = 2,
    n_dim = 144,
    keep_prob = 0.9,
    pathways = [
        range(0, 144),
        range(0, 72),
        ],
    ):
    
    training_x = [tf.placeholder(tf.float32, [None, n_dim], name='training_x' + str(n)) for n in range(n_cls)]
    x = tf.placeholder(tf.float32, [None, n_dim], name='x')
    
    training_t = [tf.placeholder(tf.float32, [len(pathways)], name='training_t' + str(n)) for n in range(n_cls)]
    training_l = [tf.placeholder(tf.float32, [len(pathways)], name='training_l' + str(n)) for n in range(n_cls)]
    
    training_mean = [tf.stack([tf.reduce_mean(tf.gather(tf.transpose(training_x[n]), pathways[p])) for p in range(len(pathways))]) for n in range(n_cls)]
    mean = tf.stack([tf.reduce_mean(tf.gather(tf.transpose(x), pathways[p])) for p in range(len(pathways))])

    a = tf.Variable(tf.random_uniform([1],
                    - 1.0 / math.sqrt(1),
                    1.0 / math.sqrt(1)))
    
    b = tf.Variable(tf.random_uniform([n_cls],
                    - 1.0 / math.sqrt(n_cls),
                    1.0 / math.sqrt(n_cls)))
    
    training_g = []
    for n in range(n_cls):
        training_g_nom = tf.nn.sigmoid(a*(training_mean[n]-b)) - tf.nn.sigmoid(-a*b)
        training_g_den = tf.nn.sigmoid(a*(1.0-b)) - tf.nn.sigmoid(-a*b)
        training_g_den = 1e-10 if training_g_den == 0.0 else training_g_den
        training_g.append(training_g_nom / training_g_den)    
    
    g_nom = tf.nn.sigmoid(a*(mean-b)) - tf.nn.sigmoid(-a*b)
    g_den = tf.nn.sigmoid(a*(1.0-b)) - tf.nn.sigmoid(-a*b)
    
    g_den = 1e-10 if g_den == 0.0 else g_den
    g = g_nom / g_den
    
    cost_na = tf.zeros([1])
    
    for n in range(n_cls):
        cost_na += -tf.reduce_mean(training_t[n]*tf.log(tf.clip_by_value(training_g[n], 1e-10, 1.0))+(1.0-training_t[n])*tf.log(tf.clip_by_value(1.0-training_g[n], 1e-10, 1.0)))
        
    cost_na /= n_cls
    
    weight = tf.Variable(tf.random_uniform([n_cls, n_cls],
                    - 1.0 / math.sqrt(n_cls),
                    1.0 / math.sqrt(n_cls)))
    
    bias = tf.Variable(tf.random_uniform([n_cls],
                    - 1.0 / math.sqrt(n_cls),
                    1.0 / math.sqrt(n_cls)))
    
    training_g_dropout = [tf.nn.dropout(training_g[n], keep_prob) for n in range(n_cls)]
    
    training_y = [tf.matmul(tf.reshape(training_g_dropout[n], [1, n_cls]), weight) + bias for n in range(n_cls)]
    y = tf.nn.softmax(tf.matmul(tf.reshape(g, [1, n_cls]), weight) + bias)

    prediction = tf.argmax(y, 1)
    
    cost_fc = tf.zeros([1])
    
    for n in range(n_cls):
        cost_fc += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=training_l[n], logits=tf.clip_by_value(training_y[n], 1e-10, 1.0)))
             
    cost_fc /= n_cls
    
    cost = cost_na + cost_fc
        
    return {
        'cost': cost,
        'training_x': training_x,
        'training_t': training_t,
        'training_l': training_l,
        'x': x,
        'y': y,
        'g': g,
        'prediction': prediction,
        'a': a,
        'b': b,
        'weight': weight,
        'bias': bias,        
    } 
    