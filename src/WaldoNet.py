import tensorflow as tf

_REGULARIZATION_SCALE = 0.01
regularizer = tf.contrib.layers.l2_regularizer(scale=_REGULARIZATION_SCALE)
initializer = tf.variance_scaling_initializer()

def inference(x, num_classes, batch_size):
    conv25 = tf.layers.conv2d(x,
                              filters=2,
                              kernel_size=[25,25],
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              use_bias=False,
                              name='layer1/conv25')
    conv15 = tf.layers.conv2d(x,
                              filters=2,
                              kernel_size=[15,15],
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              use_bias=False,
                              name='layer1/conv15')
    conv10 = tf.layers.conv2d(x,
                              filters=4,
                              kernel_size=[10,10],
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              use_bias=False,
                              name='layer1/conv10')
    conv7 = tf.layers.conv2d(x,
                              filters=8,
                              kernel_size=[7,7],
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              use_bias=False,
                              name='layer1/conv7')
    conv5 = tf.layers.conv2d(x,
                              filters=16,
                              kernel_size=[5,5],
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              use_bias=False,
                              name='layer1/conv5')
    conv3 = tf.layers.conv2d(x,
                              filters=32,
                              kernel_size=[3,3],
                              padding='same',
                              activation=tf.nn.relu,
                              kernel_initializer=initializer,
                              kernel_regularizer=regularizer,
                              use_bias=False,
                              name='layer1/conv3')
    layer1_out = tf.concat([x, conv25, conv15, conv10, conv7, conv5, conv3], axis=3, name="layer1/concat")
    layer2_conv = tf.layers.conv2d(layer1_out,
                                   filters=64,
                                   kernel_size=[3,3],
                                   padding='same',
                                   activation=tf.nn.tanh,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=regularizer,
                                   use_bias=False,
                                   name='layer2/conv')
    probability_map = tf.layers.conv2d(layer2_conv,
                                       filters=1,
                                       kernel_size=[1,1],
                                       padding='same',
                                       kernel_initializer=initializer,
                                       kernel_regularizer=regularizer,
                                       use_bias=False,
                                       name='probabililty_map')
    pooled_prob = tf.layers.max_pooling2d(probability_map,
                                              pool_size=[50,50],
                                              strides=[50,50],
                                              name='pooled')
    linear = tf.reshape(pooled_prob, [batch_size, num_classes], 'output_1d')
    return linear

def loss(inference, location_one_hot):
    probabilities = tf.nn.softmax(inference, name='output_softmax')
    cross_entropy = tf.losses.softmax_cross_entropy(probabilities, location_one_hot)
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    reg_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = tf.contrib.layers.apply_regularization(regularizer, reg_vars)
    loss_ = cross_entropy+reg_loss
    return loss_

