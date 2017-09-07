import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import cPickle as pickle
import numpy as np
import tensorflow as tf


reviews_list, sentiments_vector = pickle.load(open("imdb_train_data.pkl", "rb"))

max_review_length = max([len(x.split(" ")) for x in reviews_list])
print max_review_length

vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_review_length)

dataX = np.array(list(vocab_processor.fit_transform(reviews_list)))#dataX contains the index of each word, instead of full word.
dataY = sentiments_vector
vocab_size = len(vocab_processor.vocabulary_)



sample_index = np.random.permutation(dataX.shape[0])
num_of_testdata = int(dataX.shape[0] * 0.1) #10% of all data will be test data

trainX = np.array([dataX[i] for i in sample_index[num_of_testdata :]])
trainY = np.array([dataY[i] for i in sample_index[num_of_testdata :]])

testX = np.array([dataX[i] for i in sample_index[ : num_of_testdata]])
testY = np.array([dataY[i] for i in sample_index[ : num_of_testdata]])

print trainX.shape
print testX.shape

del dataX
del dataY

# Network building
net = tflearn.input_data([None, max_review_length])
net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.5)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.01,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(trainX, trainY,validation_set=(testX,testY), show_metric=True, batch_size=64, n_epoch=2)

