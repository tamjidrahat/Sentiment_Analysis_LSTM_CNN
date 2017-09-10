import tensorflow as tf
import os
import datetime
from text_LSTM import TextLSTM
import cPickle as pickle
import numpy as np


##### Parameters ###################################


# Model Hyperparameters
learning_rate = 0.001
embedding_dim = 128
dropout_rate = 0.5

# Training parameters
batch_size = 64
num_epochs = 30
lstm_units = 10
evaluate_every = 60
checkpoint_every = 100
num_checkpoints = 5


allow_soft_placement = True
log_device_placement = False







### DATA PROCESSING ######################################

reviews_list, sentiments_vector = pickle.load(open("imdb_train_data.pkl", "rb"))

max_review_length = max([len(x.split(" ")) for x in reviews_list])


vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_review_length)

dataX = np.array(list(vocab_processor.fit_transform(reviews_list)))#dataX contains the index of each word, instead of full word.
dataY = sentiments_vector
vocab_size = len(vocab_processor.vocabulary_)



sample_index = np.random.permutation(dataX.shape[0])
num_of_testdata = int(dataX.shape[0] * 0.2) #20% of all data will be test data

trainX = np.array([dataX[i] for i in sample_index[num_of_testdata :]])
trainY = np.array([dataY[i] for i in sample_index[num_of_testdata :]])

testX = np.array([dataX[i] for i in sample_index[ : num_of_testdata]])
testY = np.array([dataY[i] for i in sample_index[ : num_of_testdata]])

print "Train set: "+str(trainX.shape)
print "Test Shape: "+str(testX.shape)
print "Vocab Size: "+ str(vocab_size)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]




##### Training  ###################################

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement = allow_soft_placement,
      log_device_placement = log_device_placement)

    sess = tf.Session(config=session_conf)

    with sess.as_default():
        lstm = TextLSTM(
            sequence_length= trainX.shape[1],
            num_classes= trainY.shape[1],
            vocab_size= vocab_size,
            embedding_size=embedding_dim,
            lstm_units=lstm_units)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)

        optimizer = tf.train.AdamOptimizer(learning_rate) #learning rate = 0.001
        grads_and_vars = optimizer.compute_gradients(lstm.loss) #it returns a list of gradients and variables. W, dW, b, db etc.

        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)   #updates variable
        #train_op = optimizer.minimize(cnn.loss, global_step=global_step)
        # Keep track of gradient values and sparsity (optional)




        # Initialize all variables
        sess.run(tf.global_variables_initializer())


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              lstm.input_x: x_batch,
              lstm.input_y: y_batch,
              lstm.dropout_keep_prob: dropout_rate
            }
            _, step, loss, accuracy= sess.run(
                [train_op, global_step, lstm.loss, lstm.accuracy],
                feed_dict)
            #accuracy = tf.metrics.accuracy(labels=cnn.input_y, predictions=cnn.classes)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}, acc {}".format(time_str, step, loss, accuracy))


        def dev_step(x_batch, y_batch):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              lstm.input_x: x_batch,
              lstm.input_y: y_batch,
              lstm.dropout_keep_prob: 1.0 #dropout is disabled during testing
            }
            step, loss, accuracy= sess.run(
                [global_step, lstm.loss, lstm.accuracy],
                feed_dict)
            #accuracy = tf.metrics.accuracy(labels=cnn.input_y, predictions=cnn.classes)
            time_str = datetime.datetime.now().isoformat()
            print("Evaluation: {}: step {}, loss {}, acc {}".format(time_str, step, loss, accuracy))


        # Generate batches
        batches = batch_iter(list(zip(trainX, trainY)), batch_size, num_epochs)
        # Training loop. For each batch...

        for batch in batches:
            x_batch, y_batch = zip(*batch)

            train_step(x_batch, y_batch)

            current_step = tf.train.global_step(sess, global_step)

            if current_step % evaluate_every == 0:
                dev_step(testX, testY)




