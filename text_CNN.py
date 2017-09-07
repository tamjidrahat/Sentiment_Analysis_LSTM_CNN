import tensorflow as tf

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # dropout is input here. bcz we don't use it during test time
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        ##### Embedding layer ########

        #Force tensorflow to use CPU.
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # W is our embedding matrix that we learn during Training
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")

            #tf.nn.embedding_lookup() does the embedding operation. returns [None, sequence_length, embedding_size] matrix
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)

            #Tensorflow Convolution expects 4-D Tensor. add 1 in the last position of the shape.
            # Dim= [None, sequence_length, embedding_size, 1]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size.
        # We use filters of different size. So, Convolution produces tensors of different shape.
        # We store them in pooled_outputs[] list
        pooled_outputs = []

        for i, filter_size in enumerate(filter_sizes):  #for each filter size, ex: 2, 3, 4

            with tf.name_scope("conv-maxpool-%s" % filter_size):
                print "embedded chars shape: "+ str(self.embedded_chars_expanded.shape)
                print "embedding size: "+str(embedding_size)

                conv1 = tf.layers.conv2d(self.embedded_chars_expanded
                                         ,filters=num_filters
                                         ,kernel_size=[filter_size, embedding_size]
                                         ,strides=(1,1)
                                         ,padding='valid'
                                         ,activation=tf.nn.relu
                                         )



                pool1 = tf.layers.max_pooling2d(conv1
                                                ,pool_size=[sequence_length - filter_size + 1, 1]
                                                ,strides=1
                                                ,padding='valid'
                                                ,name='pool1')
                pooled_outputs.append(pool1)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)

        #h_pool is a vector of dimension num_filters_total
        self.h_pool = tf.concat(pooled_outputs, 3)      #concat all pooled tensors of pooled_outputs in axis=3 dimension.

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total]) #   reshape into [batch_size, num_filters_total]


        dropout = tf.layers.dropout(self.h_pool_flat, rate= 1.- self.dropout_keep_prob)
        logits = tf.layers.dense(dropout, 2)


        self.classes = tf.argmax(input=logits, axis=1),
        self.probabilities = tf.nn.softmax(logits)

        self.loss = tf.losses.softmax_cross_entropy(self.input_y, logits)

        correct_predictions = tf.equal(self.classes, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

