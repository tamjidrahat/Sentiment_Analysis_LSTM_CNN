import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import cPickle as pickle
import numpy as np


max_sentence_length = 500


train, val, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train
valX, valY = val

print len(trainX)
print len(valX)

for t in trainX:
    print len(t)


# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=max_sentence_length, value=0.)
valX = pad_sequences(valX, maxlen=max_sentence_length, value=0.)
# 1-Hot encoding
trainY = to_categorical(trainY, nb_classes=2)#1-Hot encoding
valY = to_categorical(valY, nb_classes=2)#1-Hot encoding

# Network building
net = tflearn.input_data([None, max_sentence_length])
net = tflearn.embedding(net, input_dim=10000, output_dim=256)
net = tflearn.lstm(net, 256, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)

model.fit(trainX, trainY, validation_set=(valX, valY), show_metric=True, batch_size=64, n_epoch=2)

