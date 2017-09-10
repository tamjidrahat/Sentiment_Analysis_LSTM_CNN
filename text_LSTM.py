import tensorflow as tf

class TextLSTM(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, lstm_units):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

        # dropout is input here. bcz we don't use it during test time
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


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

            #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        print self.embedded_chars.shape

        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_units)
        lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=0.75)
        rnn, _ = tf.nn.dynamic_rnn(lstm, self.embedded_chars, dtype=tf.float32)

        #print "rnn shape: "+str(rnn.shape)
        dense = tf.layers.dense(rnn, 2)

        #print "logits shape: " + str(dense.shape)

        value = tf.transpose(dense, [1, 0, 2])
        #print "value shape: " + str(value.shape)

        output = tf.gather(value, int(value.get_shape()[0]) - 1)

        #print "last shape: " + str(output.shape)

        self.classes = tf.argmax(input=output, axis=1),
        self.probabilities = tf.nn.softmax(output)

        self.loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(self.input_y, output))

        correct_predictions = tf.equal(self.classes, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

