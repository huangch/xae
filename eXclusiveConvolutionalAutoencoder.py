'''eXclusive Autoencoder

Huang et al., Dec 2018
'''
import tensorflow as tf
import numpy as np
import math
# 
# def exclusive_pool_with_argmax(value, ksize, strides, padding, n_channels, pathways):
#     """
#     Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
#     Therefore, we use max_pool_with_argmax to extract mask and
#     plain max_pool for, eeem... max_pooling.
#     """
# 
#     with tf.name_scope('eXclusivePoolArgMax'):
#         _, inc_argmax_mask = tf.nn.max_pool_with_argmax(
#             value,
#             ksize=ksize,
#             strides=strides,
#             padding=padding)
# 
#         inc_argmax_mask = tf.stop_gradient(inc_argmax_mask)
#                   
#         inc_value = tf.nn.max_pool(
#           value,
#           ksize=ksize,
#           strides=strides,
#           padding=padding)
#         
#         exc_pathways = np.setdiff1d(range(n_channels), pathways).tolist()
#         exc_value = tf.reciprocal(tf.clip_by_value(value, clip_value_min=1e-10, clip_value_max=1e10))
#         
#         _, exc_argmax_mask = tf.nn.max_pool_with_argmax(
#             exc_value,
#             ksize=ksize,
#             strides=strides,
#             padding=padding)
#         
#         exc_argmax_mask = tf.stop_gradient(exc_argmax_mask)
#         
#         exc_value = tf.nn.max_pool(
#           exc_value,
#           ksize=ksize,
#           strides=strides,
#           padding=padding)     
#         
#         exc_value = tf.reciprocal(exc_value)
#         
#               
#         tf.scatter_nd_update()
#         
#         
#         ??? 
#       
#     return value, argmax_mask
#               
def max_pool_with_argmax(value, ksize, strides, padding):
    """
    Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
    Therefore, we use max_pool_with_argmax to extract mask and
    plain max_pool for, eeem... max_pooling.
    """
#         inc_channels = pathway
#         exc_channels = [c for c in range(n_channels) if c not in inc_channels]
#         
#         inc_value = tf.gather(value, inc_channels, axis=3)        
#         exc_value = tf.gather(value, exc_channels, axis=3)
#         
#         inc_value, inc_argmax_mask = tf.nn.max_pool_with_argmax(
#             inc_value,
#             ksize=ksize,
#             strides=strides,
#             padding=padding)
#         
#         exc_value, exc_argmax_mask = tf.nn.max_pool_with_argmax(
#             tf.reciprocal(tf.clip_by_value(exc_value, clip_value_min=1e-10, clip_value_max=1e10)),
#             ksize=ksize,
#             strides=strides,
#             padding=padding)                
#         
#         exc_value = tf.reciprocal(tf.clip_by_value(exc_value, clip_value_min=1e-10, clip_value_max=1e10))




        
        
    with tf.name_scope('MaxPoolArgMax'):        
        value, argmax_mask = tf.nn.max_pool_with_argmax(
            value,
            ksize=ksize,
            strides=strides,
            padding=padding)
        
#         argmax_mask = tf.stop_gradient(argmax_mask)
#                   
#         value = tf.nn.max_pool(
#           value,
#           ksize=ksize,
#           strides=strides,
#           padding=padding)
      
    return value, argmax_mask


# Thank you, @https://github.com/Pepslee
def unpool_with_argmax(value, argmax_mask, ksize):
    assert argmax_mask is not None
    with tf.name_scope('UnPool2D'):
    
        input_shape = value.get_shape().as_list()
        #  calculation new shape
        output_shape = (tf.cast(tf.shape(value)[0], tf.int64), input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
        # calculation indices for batch, height, width and feature maps
        one_like_mask = tf.ones_like(argmax_mask, dtype=tf.int64)
        batch_range = tf.reshape(tf.range(tf.cast(tf.shape(value)[0], tf.int64), dtype=tf.int64), shape=[tf.shape(value)[0], 1, 1, 1])
        b = one_like_mask * batch_range
        y = argmax_mask // (output_shape[2] * output_shape[3])
        x = argmax_mask % (output_shape[2] * output_shape[3]) // output_shape[3]
        feature_range = tf.range(output_shape[3], dtype=tf.int64)
        f = one_like_mask * feature_range
        # transpose indices & reshape update values to one dimension
        updates_size = tf.size(value)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(value, [updates_size])
        ret = tf.scatter_nd(indices, values, output_shape)
        return ret

# %%
def eXclusiveConvolutionalAutoencoder(
        input_shape = [None, 28, 28, 1],
         
        layers = [
            {
                'n_channels': 144,
                'reconstructive_regularizer': 1.0, 
                'weight_decay': 1.0, 
                'sparse_regularizer': 1.0, 
                'sparsity_level': 0.05,
                'exclusive_regularizer': 1.0,
                'exclusive_scale': 1.0,
                'tied_weight': True,
                'conv_size': 8,
                'conv_stride': 1,
                'conv_padding': 'VALID',
                # 'pool_size': 0,
                # 'pool_stride': 0,
                # 'pool_padding': 'VALID',
                'corrupt_prob': 0.5,
                'exclusive_type': 'logcosh',
                'exclusive_scale': 1.0,    
                # 'gaussian_mean': 0.0,    
                # 'gaussian_std': 0.0,                                
                'encode':'sigmoid', 'decode':'linear',
                'pathways': [
                    range(0, 72),
                    range(0, 144),
                ],
            },                                                                                                 
        ],
        
        init_encoder_weight = None,
        init_decoder_weight = None,
        init_encoder_bias = None,
        init_decoder_bias = None,                
    ):
    '''Build a deep denoising autoencoder w/ tied weights.
 
    Parameters
    ----------
    input_shape : list, optional
        Description
    n_channels : list, optional
        Description
    filter_sizes : list, optional
        Description
 
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
 
    Raises
    ------
    ValueError
        Description
    '''
    # %%
    
    n_channels = [input_shape[3]]
    
    for layer in layers:
        n_channels.append(layer['n_channels'])
        assert len(layer['pathways']) == len(layers[0]['pathways']), 'Ambiguous pathway definitions over layers.'
                  
    # %% input to the network
    training_x = []
    training_x_tensor = []
    for pathway_i in range(len(layers[0]['pathways'])):
        # ensure 2-d is converted to square tensor.
        x = tf.placeholder(tf.float32, input_shape, name='training_x' + str(pathway_i))
        training_x.append(x)
        if len(x.get_shape()) == 2:
            x_dim = np.sqrt(x.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(
                x, [-1, x_dim, x_dim, n_channels[0]])
        elif len(x.get_shape()) == 4:
            x_tensor = x
        else:
            raise ValueError('Unsupported input dimensions')
         
        training_x_tensor.append(x_tensor)
         
    # input to the network
    x = tf.placeholder(tf.float32, input_shape, name='x')
 
    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_channels[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
     
    # current_input = x
    current_input = x_tensor

    training_current_input = []
    for pathway_i in range(len(layers[0]['pathways'])):
        training_current_input.append(training_x_tensor[pathway_i])

    # %%
    # Build the encoder
    encoder_weight = []
    encoder_bias = []
    training_shape_list = []
    shape_list = []
    training_encoder_output_list = []
    training_encoder_input_list = []
    layerwise_z = []
    
    training_argmax_list = []
    argmax_list = []
    
    for layer_i, (n_input, n_output) in enumerate(zip(n_channels[:-1], n_channels[1:])):
        training_shape_list.append([])
        shape_list.append(current_input.get_shape().as_list())
        
        if init_encoder_weight != None:
            W = tf.Variable(tf.constant(init_encoder_weight[layer_i]))            
        else:
            W = tf.Variable(
                tf.random_uniform([
                    layers[layer_i]['conv_size'],
                    layers[layer_i]['conv_size'],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
        
        if init_encoder_bias != None:
            b = tf.Variable(tf.constant(init_encoder_bias[layer_i]))
        else:
            b = tf.Variable(tf.zeros([n_output]))
        
        encoder_weight.append(W)
        encoder_bias.append(b)
        
        if 'pool_size' in layers[layer_i] and 'pool_stride' in layers[layer_i] and 'pool_padding' in layers[layer_i]:
            training_argmax_list.append([])            
                
        training_encoder_input_list.append([]) 
        training_encoder_output = []
        
        for pathway_i in range(len(layers[layer_i]['pathways'])):
            training_encoder_input_list[-1].append(corrupt(training_current_input[pathway_i]) * layers[layer_i]['corrupt_prob'] + training_current_input[pathway_i] * (1 - layers[layer_i]['corrupt_prob']) if layers[layer_i]['corrupt_prob'] != None else training_current_input[pathway_i])            
            training_shape_list[-1].append(training_current_input[pathway_i]._shape_as_list())
            
            a = activate_function(
                tf.add(
                    tf.nn.conv2d(
                        training_current_input[pathway_i], 
                        W, 
                        strides=[1, layers[layer_i]['conv_stride'], layers[layer_i]['conv_stride'], 1],
                        padding=layers[layer_i]['conv_padding'],
                        ), 
                    b
                    ), 
                layers[layer_i]['encode']
                )       
            
            if 'pool_size' in layers[layer_i] and 'pool_stride' in layers[layer_i] and 'pool_padding' in layers[layer_i]:
                a, argmax = max_pool_with_argmax(
                    a, 
                    ksize = [1, layers[layer_i]['pool_size'], layers[layer_i]['pool_size'], 1],
                    strides = [1, layers[layer_i]['pool_stride'], layers[layer_i]['pool_stride'], 1],
                    padding = layers[layer_i]['pool_padding']
                    )
                 
                training_argmax_list[-1].append(argmax)
            
            training_encoder_output.append(a)
            
        training_encoder_output_list.append(training_encoder_output)
        training_current_input = training_encoder_output  
        
        output = activate_function(
            tf.add(
                tf.nn.conv2d(
                    current_input, 
                    W, 
                    strides=[1, layers[layer_i]['conv_stride'], layers[layer_i]['conv_stride'], 1], 
                    padding=layers[layer_i]['conv_padding']), 
                b), 
            layers[layer_i]['encode'])
        
        if 'pool_size' in layers[layer_i] and 'pool_stride' in layers[layer_i] and 'pool_padding' in layers[layer_i]:
            output, argmax = max_pool_with_argmax(
                output, 
                ksize = [1, layers[layer_i]['pool_size'], layers[layer_i]['pool_size'], 1],
                strides = [1, layers[layer_i]['pool_stride'], layers[layer_i]['pool_stride'], 1],
                padding = layers[layer_i]['pool_padding']
                )
             
            argmax_list.append(argmax)
            
        layerwise_z.append(output)            
        current_input = output
        
    # %% latent representation
    training_z = training_encoder_output
    z = current_input
     
    decoder_weight = []
    decoder_bias = []
    
    training_decoder_output_list = []
    
    # layerwise_training_decoder_input_list = []
    layerwise_training_decoder_output_list = []
    layerwise_y = []
    
    # %% Build the decoder using the same weights
    for layer_i, (n_input, n_output, training_shape, shape) in enumerate(zip(n_channels[::-1][:-1], n_channels[::-1][1:], training_shape_list[::-1], shape_list[::-1])):
        if init_decoder_weight != None:
            W = tf.Variable(tf.constant(init_decoder_weight[::-1][layer_i]))            
        else:
            if layers[layer_i]['tied_weight'] == True:
                W = encoder_weight[::-1][layer_i]
            else:
                W = tf.Variable(
                tf.random_uniform([
                    layers[::-1][layer_i]['conv_size'],
                    layers[::-1][layer_i]['conv_size'],
                    n_output, n_input],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input))) 
        
        if init_decoder_bias != None:
            b = tf.Variable(tf.constant(init_decoder_bias[::-1][layer_i]))
        else:
            b = tf.Variable(tf.zeros([n_output]))
        
        decoder_weight.append(W)
        decoder_bias.append(b)
         
        training_decoder_output = []
        
        # layerwise_training_decoder_input_list.append([])
        layerwise_training_decoder_output = []

        for pathway_i in range(len(layers[::-1][layer_i]['pathways'])):
            
            # layerwise_training_decoder_input_list[-1].append(tf.identity(training_encoder_output_list[::-1][layer_i][pathway_i]))
            
            training_current_input_pathway_i = training_current_input[pathway_i]
            
            if 'pool_size' in layers[layer_i] and 'pool_stride' in layers[layer_i] and 'pool_padding' in layers[layer_i]:
                training_current_input_pathway_i = unpool_with_argmax(
                    training_current_input_pathway_i, 
                    training_argmax_list[::-1][layer_i][pathway_i],         
                    ksize = [1, layers[::-1][layer_i]['pool_size'], layers[::-1][layer_i]['pool_size'], 1]
                    )
        
            a = activate_function(
                tf.add(
                    tf.nn.conv2d_transpose(
                        tf.gather(training_current_input_pathway_i, layers[::-1][layer_i]['pathways'][pathway_i], axis=3),
                        # training_current_input[pathway_i],
                        tf.gather(W, layers[::-1][layer_i]['pathways'][pathway_i], axis=3),
                        # W,
                        tf.stack([
                            tf.shape(training_current_input[pathway_i])[0],
                            training_shape[pathway_i][1], 
                            training_shape[pathway_i][2], 
                            n_output,
                            ]),
                        strides=[1, layers[::-1][layer_i]['conv_stride'], layers[::-1][layer_i]['conv_stride'], 1], 
                        padding=layers[::-1][layer_i]['conv_padding'],
                        ), 
                    b
                    ), 
                layers[::-1][layer_i]['decode'])
            
            training_decoder_output.append(a)
            
            
            training_encoder_output_list_rev_layer_i_pathway_i = training_encoder_output_list[::-1][layer_i][pathway_i]
            
            if 'pool_size' in layers[layer_i] and 'pool_stride' in layers[layer_i] and 'pool_padding' in layers[layer_i]:
                # layerwise_training_decoder_input_list[-1][-1] = unpool_with_argmax(
                training_encoder_output_list_rev_layer_i_pathway_i = unpool_with_argmax(
                    # layerwise_training_decoder_input_list[-1][-1],
                    training_encoder_output_list_rev_layer_i_pathway_i,
                    training_argmax_list[::-1][layer_i][pathway_i],         
                    ksize = [1, layers[::-1][layer_i]['pool_size'], layers[::-1][layer_i]['pool_size'], 1]
                    )
                          
            layerwise_a = activate_function(
                tf.add(
                    tf.nn.conv2d_transpose(
                        # tf.gather(layerwise_training_decoder_input_list[-1][-1], layers[::-1][layer_i]['pathways'][pathway_i], axis=3),
                        tf.gather(training_encoder_output_list_rev_layer_i_pathway_i, layers[::-1][layer_i]['pathways'][pathway_i], axis=3),
                        # layerwise_training_decoder_input_list[-1][-1],
                        tf.gather(W, layers[::-1][layer_i]['pathways'][pathway_i], axis=3),                        
                        # W,
                        tf.stack([
                            tf.shape(training_encoder_output_list[::-1][layer_i][pathway_i])[0],
                            training_shape[pathway_i][1], 
                            training_shape[pathway_i][2], 
                            n_output,
                            ]),
                        strides=[1, layers[::-1][layer_i]['conv_stride'], layers[::-1][layer_i]['conv_stride'], 1], 
                        padding=layers[::-1][layer_i]['conv_padding'],
                        ), 
                    b
                    ), 
                layers[::-1][layer_i]['decode'])

            layerwise_training_decoder_output.append(layerwise_a)

        training_decoder_output_list.append(training_decoder_output)  
        layerwise_training_decoder_output_list.append(layerwise_training_decoder_output) 
        training_current_input = training_decoder_output                

        if 'pool_size' in layers[layer_i] and 'pool_stride' in layers[layer_i] and 'pool_padding' in layers[layer_i]:
            current_input = unpool_with_argmax(
                current_input, 
                argmax_list[::-1][layer_i],         
                ksize = [1, layers[::-1][layer_i]['pool_size'], layers[::-1][layer_i]['pool_size'], 1]
                )
          
        output = activate_function(
            tf.add(
                tf.nn.conv2d_transpose(
                    current_input, 
                    W,
                    tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                    strides=[1, layers[::-1][layer_i]['conv_stride'], layers[::-1][layer_i]['conv_stride'], 1], 
                    padding=layers[::-1][layer_i]['conv_padding']
                ), 
                b
            ), 
            layers[::-1][layer_i]['decode']
        )
                                          
        current_input = output

    for layer_i in range(len(layers)):
        layerwise_input = layerwise_z[::-1][layer_i]
        for layer_j in range(len(layers)-layer_i)[::-1]:
            if 'pool_size' in layers[layer_j] and 'pool_stride' in layers[layer_j] and 'pool_padding' in layers[layer_j]:
                layerwise_input = unpool_with_argmax(
                    layerwise_input, 
                    argmax_list[layer_j],         
                    ksize = [1, layers[layer_j]['pool_size'], layers[layer_j]['pool_size'], 1]
                    )
               
            layerwise_output = activate_function(
                tf.add(
                    tf.nn.conv2d_transpose(
                        layerwise_input, 
                        decoder_weight[::-1][layer_j],
                        tf.stack([tf.shape(x)[0], shape_list[layer_j][1], shape_list[layer_j][2], shape_list[layer_j][3]]),
                        strides=[1, layers[layer_j]['conv_stride'], layers[layer_j]['conv_stride'], 1], 
                        padding=layers[layer_j]['conv_padding']
                    ), 
                    decoder_bias[::-1][layer_j]
                ), 
                layers[layer_j]['decode']
            )
                     
            layerwise_input = layerwise_output
                 
        layerwise_y.append(layerwise_output)
  
    decoder_weight.reverse()
    decoder_bias.reverse()
    training_decoder_output_list.reverse()
    layerwise_training_decoder_output_list.reverse()    
    layerwise_y.reverse()

    # %% now have the reconstruction through the network
    training_y = training_current_input    
    y = current_input

    # cost function measures pixel-wise difference
    cost = {}
        
    cost['reconstruction_error'] = tf.constant(0.0)   
    
    # for layer_i in range(len(layers)):
    for pathway_i in range(len(layers[0]['pathways'])):
        if training_encoder_input_list[0][pathway_i] != None and training_decoder_output_list[0][pathway_i] != None:
            cost['reconstruction_error'] = tf.add(
                cost['reconstruction_error'], 
                layers[0]['reconstructive_regularizer'] * 0.5 * tf.reduce_mean(
                    tf.square(
                        tf.subtract(
                            training_encoder_input_list[0][pathway_i], 
                            training_decoder_output_list[0][pathway_i],
                            ))))
                 
    cost['weight_decay'] = tf.constant(0.0)   
    for layer_i in range(len(layers)):
        cost_encoder_weight_decay = layers[layer_i]['weight_decay'] * 0.5 * tf.reduce_mean(tf.square(encoder_weight[layer_i]))
        if layers[layer_i]['tied_weight']:
            cost['weight_decay'] = tf.add(cost['weight_decay'], cost_encoder_weight_decay)
        else: 
            cost_decoder_weight_decay = layers[layer_i]['weight_decay'] * 0.5 * tf.reduce_mean(tf.square(decoder_weight[layer_i]))
            cost['weight_decay'] = tf.add(cost['weight_decay'], 0.5*cost_encoder_weight_decay+0.5*cost_decoder_weight_decay)




                
    cost['exclusivity'] = tf.constant(0.0)
    for layer_i in range(len(layers)):
        for pathway_i, (encoder_pathway_output) in enumerate(training_encoder_output_list[layer_i]):  
            exclusivity = np.setdiff1d(range(layers[layer_i]['n_channels']), layers[layer_i]['pathways'][pathway_i]).tolist()   
            if exclusivity != [] and encoder_pathway_output != None:
                if layers[layer_i]['exclusive_type'] == 'pow4':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:                       
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.pow(tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity), 4),
                                )
                            )
                        )
                    else:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        tf.pow(tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity), 4),
                                        tf.pow(tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std']), 4)
                                    )
                                )
                            )
                        )                          
                elif layers[layer_i]['exclusive_type'] == 'exp':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity))),
                                )
                            )
                        )                        
                    else:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity))),
                                        (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std']))),
                                    )
                                )
                            )
                        )                                        
                elif layers[layer_i]['exclusive_type'] == 'logcosh':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity))),
                                )
                            )
                        )                        
                    else:
                        cost['exclusivity'] = tf.add(cost['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity))),
                                        (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std'])))
                                    )
                                )
                            )
                        )
    cost['sparsity'] = tf.constant(0.0)
    for layer_i in range(len(layers)):
        for pathway_i, (encoder_pathway_output) in enumerate(training_encoder_output_list[layer_i]):                
            if layers[layer_i]['pathways'][pathway_i] != None and encoder_pathway_output != None:
                cost['sparsity'] = tf.add(cost['sparsity'], layers[layer_i]['sparse_regularizer'] * tf.reduce_mean(kl_divergence(layers[layer_i]['sparsity_level'], tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), layers[layer_i]['pathways'][pathway_i]))))
                                         
    cost['total'] = cost['reconstruction_error'] + cost['weight_decay'] + cost['sparsity'] + cost['exclusivity']

    layerwise_cost = []     
    for layer_i in range(len(layers)):
        layerwise_cost.append({})
        
        layerwise_cost[layer_i]['reconstruction_error'] = tf.constant(0.0)
        for pathway_i in range(len(layers[layer_i]['pathways'])):
            if training_encoder_input_list[layer_i][pathway_i] != None and layerwise_training_decoder_output_list[layer_i][pathway_i] != None:
                layerwise_cost[layer_i]['reconstruction_error'] = tf.add(
                    layerwise_cost[layer_i]['reconstruction_error'], 
                    layers[layer_i]['reconstructive_regularizer'] * 0.5 * tf.reduce_mean(
                        tf.square(
                            tf.subtract(
                                training_encoder_input_list[layer_i][pathway_i], 
                                layerwise_training_decoder_output_list[layer_i][pathway_i],
                                ))))

                         
        layerwise_cost[layer_i]['weight_decay'] = tf.constant(0.0)         
        layerwise_cost_encoder_weight_decay = layers[layer_i]['weight_decay'] * 0.5 * tf.reduce_mean(tf.square(encoder_weight[layer_i]))
                  
        if layers[layer_i]['tied_weight']:
            layerwise_cost[layer_i]['weight_decay'] = tf.add(layerwise_cost[layer_i]['weight_decay'], layerwise_cost_encoder_weight_decay)
        else:
            layerwise_cost_decoder_weight_decay = layers[layer_i]['weight_decay'] * 0.5 * tf.reduce_mean(tf.square(decoder_weight[layer_i]))
            layerwise_cost[layer_i]['weight_decay'] = tf.add(layerwise_cost[layer_i]['weight_decay'], 0.5*layerwise_cost_encoder_weight_decay+0.5*layerwise_cost_decoder_weight_decay)            
               
        layerwise_cost[layer_i]['exclusivity'] = tf.constant(0.0) 
        for pathway_i, (encoder_pathway_output) in enumerate(training_encoder_output_list[layer_i]):                
            exclusivity = np.setdiff1d(range(layers[layer_i]['n_channels']), layers[layer_i]['pathways'][pathway_i]).tolist()
              
            if exclusivity != [] and encoder_pathway_output != None:
                if layers[layer_i]['exclusive_type'] == 'pow4':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.pow(tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity), 4),
                                )
                            )
                        )
                    else:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        tf.pow(tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity), 4),
                                        tf.pow(tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std']), 4)
                                    )
                                )
                            )
                        )
                elif layers[layer_i]['exclusive_type'] == 'exp':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity))),
                                )
                            )
                        )                        
                    else:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity))),
                                        (-1/layers[layer_i]['exclusive_scale'])*tf.exp(-0.5*layers[layer_i]['exclusive_scale']*tf.square(tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std'])))
                                    )
                                )
                            )
                        )
                elif layers[layer_i]['exclusive_type'] == 'logcosh':
                    if not 'gaussian_mean' in layers[layer_i] or not 'gaussian_std' in layers[layer_i]:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity))),                                                                         
                                )
                            )
                        )
                    else:
                        layerwise_cost[layer_i]['exclusivity'] = tf.add(layerwise_cost[layer_i]['exclusivity'], 
                            layers[layer_i]['exclusive_regularizer'] * 0.5 * tf.square(
                                tf.reduce_mean(
                                    tf.subtract(
                                        (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), exclusivity))),
                                        (1/layers[layer_i]['exclusive_scale'])*tf.log(tf.cosh(layers[layer_i]['exclusive_scale']*tf.random_normal([len(exclusivity)], mean=layers[layer_i]['gaussian_mean'], stddev=layers[layer_i]['gaussian_std'])))
                                    )                                    
                                )
                            )
                        )                                                

        layerwise_cost[layer_i]['sparsity'] = tf.constant(0.0)
        for pathway_i, (encoder_pathway_output) in enumerate(training_encoder_output_list[layer_i]):                
            if layers[layer_i]['pathways'][pathway_i] != None and encoder_pathway_output != None:
                layerwise_cost[layer_i]['sparsity'] = tf.add(layerwise_cost[layer_i]['sparsity'], layers[layer_i]['sparse_regularizer'] * tf.reduce_mean(kl_divergence(layers[layer_i]['sparsity_level'], tf.gather(tf.reduce_mean(encoder_pathway_output, [0,1,2]), layers[layer_i]['pathways'][pathway_i]))))
                                             
        layerwise_cost[layer_i]['total'] = layerwise_cost[layer_i]['reconstruction_error'] + layerwise_cost[layer_i]['weight_decay'] + layerwise_cost[layer_i]['exclusivity'] + layerwise_cost[layer_i]['sparsity']
        
    # %%
    return {'training_x': training_x, 'training_z': training_z, 'training_y': training_y,
            'x': x, 'y': y, 'z': z, 
            'layerwise_y': layerwise_y, 'layerwise_z': layerwise_z, 
            'cost': cost,
            'layerwise_cost': layerwise_cost, 
            'encoder_weight': encoder_weight, 'decoder_weight': decoder_weight,
            'encoder_bias': encoder_bias, 'decoder_bias': decoder_bias,
        }

def kl_divergence(p, p_hat):
    return p * tf.log(tf.clip_by_value(p, 1e-10, 1e10)) - p * tf.log(tf.clip_by_value(p_hat, 1e-10, 1e10)) + (1 - p) * tf.log(tf.clip_by_value(1 - p, 1e-10, 1e10)) - (1 - p) * tf.log(tf.clip_by_value(1 - p_hat, 1e-10, 1e10))
    
# %%
def corrupt(x):
    '''Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.

    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    '''
    return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                               minval=0,
                                               maxval=2,
                                               dtype=tf.int32), tf.float32))

def activate_function(linear, name, leak=0.2):
    if name == 'sigmoid':
        return tf.nn.sigmoid(linear, name='encoded')
    elif name == 'softmax':
        return tf.nn.softmax(linear, name='encoded')
    elif name == 'linear':
        return linear
    elif name == 'tanh':
        return tf.nn.tanh(linear, name='encoded')
    elif name == 'relu':
        return tf.nn.relu(linear, name='encoded')
    elif name == 'lrelu':
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * linear + f2 * abs(linear)

def get_scope_variable(scope_name, var, shape=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v