import pandas as pd
from bs4 import BeautifulSoup
import nltk
import re
import cPickle as pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer



def process_review(review):
    clean_text = BeautifulSoup(review, "html5lib").get_text()   #remove html tags
    letter_only = re.sub("[^a-zA-Z]", " ", clean_text)  #remove everything except letters
    letter_only = letter_only.lower()   #convert to lowercase
    words = letter_only.split()     #split into words
    words = [w for w in words if w not in nltk.corpus.stopwords.words('english')]   #remove stopwords

    return " ".join(words)  #join the words back into full sentence

def process_sentiments(y):
    y = np.asarray(y, dtype='int32')

    Y = np.zeros((len(y), 2))   #2 classes
    Y[np.arange(len(y)), y] = 1.
    return Y

def read_dataset(filepath):
    traindata = pd.read_csv(filepath, compression='zip',
                            header=0, delimiter='\t', quoting=3)

    train_reviews = [process_review(review) for review in traindata['review']]  #clean all reviews dim = (m x num_features)
    train_sentiments = process_sentiments(traindata['sentiment'])   #(m x num_classes)

    return train_reviews, train_sentiments

def get_bag_of_words_train(train_reviews, num_of_features):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=num_of_features)
    train_data_BOW = vectorizer.fit_transform(train_reviews)       #convert each review into a vector of dimention "num_of_features"
    vocabulary = vectorizer.get_feature_names()     #get the vocabulary of "num_of_features" size
    train_data_BOW = train_data_BOW.toarray()     #convert to numpy array
    return vocabulary, train_data_BOW

def get_bag_of_words_test(train_reviews, num_of_features):
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=num_of_features)
    test_data_BOW = vectorizer.transform(train_reviews)       #called transform(). not fit_transform() like during training.
                                                                # convert each review into a vector of dimention "num_of_features"
    vocabulary = vectorizer.get_feature_names()     #get the vocabulary of "num_of_features" size
    test_data_BOW = test_data_BOW.toarray()     #convert to numpy array
    return vocabulary, test_data_BOW

if __name__ == "__main__":

    ###Process Train data###

    train_reviews, train_sentiments = read_dataset('./data/labeledTrainData.tsv.zip')

    train_data = (train_reviews, train_sentiments)
    pickle.dump(train_data, open('imdb_train_data.pkl', 'wb'))#dump train data(as list) and train label(as array)

    #train_reviews, train_sentiments = pickle.load(open('imdb_train_data.pkl', 'rb'))

    #vocabulary, train_review_BOWs = get_bag_of_words_train(train_reviews, 5000)   #return vocabulary(all words) and Bag of Words for reviews
    #train_data = (train_review_BOWs, train_sentiments)
    #pickle.dump(train_data, open('imdb_train_data_BOW.pkl', 'wb'), protocol=2)    #dump train data(as Bag of Words) and train labels
    #pickle.dump(vocabulary, open('imdb_vocabulary.pkl', 'wb'))     #dump vocabulary


    ###Process Test data###

    #test_reviews, test_sentiments = read_dataset('./data/testData.tsv.zip')
    #test_data = (test_reviews, test_sentiments)
    #pickle.dump(test_data, open('imdb_test_data.pkl', 'wb'))

    '''
    test_reviews, test_sentiments = pickle.load(open('imdb_test_data.pkl', 'rb'))
    voca, test_BOW = get_bag_of_words_test(test_reviews, 5000)
    test_data = (test_BOW, test_sentiments)
    pickle.dump(test_data, open('imdb_test_data_BOW.pkl', 'wb'), protocol=2)
    '''
    #traindata = pd.read_csv('./data/testData.tsv.zip', compression='zip', header=0, delimiter='\t', quoting=3)
    #print traindata.columns.values

