#Sentiment Analysis in IMDB movie review dataset

## ::Sentiment analysis of IMDB Movie Review dataset with CNN::
Convolution with filter size 2,3,4.
Embedding -> Convolution -> max-pooling -> dropout -> Predictions -> Loss

Train examples = 22500
Validation examples = 2500
Embedding size = 128
Filter sizes = 3,4,5
Filters of each size = 128
Dropout keep_prob = 0.5

Epochs = 200

**After 1500 steps,** 
		Train Accuracy = 100%
		Validation accuracy = 88%


![](https://github.com/tamjidrahat/Sentiment_Analysis_LSTM_CNN/blob/master/image_cnntext.png "Convolution -> Pooling")
Image courtesy:http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/





## ::Sentiment analysis of IMDB Movie Review dataset with LSTM+RNN::
LSTM units = 10
Train examples = 22500
Validation examples = 2500
Embedding size = 128


**After 200 steps,**
		Training accuracy: 50%



