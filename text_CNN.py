import tensorflow as tf
import numpy as np


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
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]    #each filter is 2D. So depth is 1
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")

                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,#  input
                    W, #    filters weight
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply ReLu
                activation = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    activation,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1], #size of window for each dimension of input
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                #after maxpooling, tensor dim = [batch_size, 1, 1, num_filters]
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)

        #h_pool is a vector of dimension num_filters_total
        self.h_pool = tf.concat(pooled_outputs, 3)      #concat all pooled tensors of pooled_outputs in axis=3 dimension.

        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total]) #   reshape into [batch_size, num_filters_total]


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores") #   xW+b
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")