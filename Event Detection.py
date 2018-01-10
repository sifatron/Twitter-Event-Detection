# Event Detection using Keras and Scikit-Learn


import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
import glove
LabeledSentence = gensim.models.doc2vec.LabeledSentence
from gensim.models import word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import logging
from gensim import corpora, models, similarities

from tqdm import tqdm

import nltk
from nltk.tokenize import TweetTokenizer 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from sklearn.linear_model import Perceptron, LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, GaussianNB

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Input, LSTM, Convolution1D, Flatten, Dropout, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import TensorBoard
from keras.utils import np_utils



def prepcsv():
    data = pd.read_csv('Name Of the File.csv')
    # data.drop(['Alternate Labels'], axis=1, inplace=True)
    data = data[data.Label.isnull() == False]
    # data['author'] = data['author'].map(int)
    data = data[data['Tweet Text'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print('dataset loaded with shape', data.shape)   
    return data

data = prepcsv()
print(data.head(2))


corpus = data['Tweet Text'].values.tolist()
data['tokens'] = [word_tokenize(sent) for sent in corpus]

print(data['tokens'][1])
data.drop(['Tweet Text'], inplace=True, axis=1)


print('dataset loaded with shape', data.shape)
print(data.head(2))

def labelizeText(text, label_type):
    labelized = []
    for i, v in enumerate(text):
        label = '%s_%s'%(label_type,i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


root_path = r"File Path"
w2vmodel = word2vec.KeyedVectors.load(root_path + "GlovE 6B Gensim VecSize300")


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0
    # vec = []
    for word in tokens:
        try:
        	# word = word.lower()
        	# if word not in stopwords:
        	# 	word = lemm.lemmatize(word)
        		# if word in w2vmodel.wv.index2word[:20000]:
        		# vec = np.concatenate((vec, w2vmodel[word]),axis = 0)
        	vec += w2vmodel[word].reshape((1, size))
            count += 1
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    # vec = sequence.pad_sequences(vec, maxlen=size)
    return vec


n=4462
vector_dim = 300
max_features = 3569
max_length = 300
embedding_dim = 100


x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).Label), test_size=0.2)


x_train = labelizeText(x_train, 'TRAIN')
x_test = labelizeText(x_test, 'TEST')
print(len(x_train))
print(x_train[4])

print('Building tf-idf matrix')
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print('vocab size :', len(tfidf))


train_vecs_w2v = np.concatenate([buildWordVector(z, max_length) for z in map(lambda x: x.words, x_train)])
train_vecs_w2v = scale(train_vecs_w2v)
print('x_train shape:', train_vecs_w2v.shape)


test_vecs_w2v = np.concatenate([buildWordVector(z, max_length) for z in map(lambda x: x.words, x_test)])
test_vecs_w2v = scale(test_vecs_w2v)
print('x_test shape:', test_vecs_w2v.shape)





print('Building Model Perceptron')
Perceptron = Perceptron(max_iter = 20000, random_state= None)
Perceptron.fit(train_vecs_w2v, y_train, coef_init=None, intercept_init=None, sample_weight=None)
score = Perceptron.score(test_vecs_w2v, y_test, sample_weight=None)
print("Accuracy fo Perceptron : %.2f%%" % (score*100))





print('Building Model Logistic')
Logistic = LogisticRegression(random_state=0, max_iter=20000)
Logistic.fit(train_vecs_w2v, y_train)
score = Logistic.score(test_vecs_w2v, y_test, sample_weight=None)
print("Accuracy fo Logistic : %.2f%%" % (score*100))





print('Building Model SGDClassifier')
SGDClassifier = SGDClassifier(loss = 'hinge', random_state=None, max_iter=20000)
SGDClassifier.fit(train_vecs_w2v, y_train)
score = SGDClassifier.score(test_vecs_w2v, y_test, sample_weight=None)
print("Accuracy fo SGDClassifier : %.2f%%" % (score*100))






print('Building Model BernoulliNB')
BernoulliNB = BernoulliNB()
BernoulliNB.fit(train_vecs_w2v, y_train)
score = BernoulliNB.score(test_vecs_w2v, y_test, sample_weight=None)
print("Accuracy fo BernoulliNB : %.2f%%" % (score*100))






print('Building Model GaussianNB')
GaussianNB = GaussianNB()
GaussianNB.fit(train_vecs_w2v, y_train)
score = GaussianNB.score(test_vecs_w2v, y_test, sample_weight=None)
print("Accuracy fo GaussianNB : %.2f%%" % (score*100))




print('Building LSTM Model')
embedding_layer = Embedding(input_dim = max_features, output_dim = embedding_dim, input_length=max_length, weights = None, trainable=True)
sequence_input = Input(shape=(max_length,), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)
lstm1 = LSTM(128)(embedded_sequences)
l_dense1 = Dense(output_dim = 64, activation='relu')(lstm1)
l_drop1 = Dropout(0.2)(l_dense1)
l_dense2 = Dense(output_dim = 32, activation='relu')(l_drop1)
l_drop2 = Dropout(0.2)(l_dense2)
preds = Dense(output_dim = 1, activation='sigmoid')(l_drop2)

model = Model(sequence_input, preds)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
# model5.load_weights('Model4_Weights.h5', by_name=True)
model.fit(train_vecs_w2v, y_train, epochs=35, batch_size=32, verbose=2)
scores= model.evaluate(test_vecs_w2v, y_test, verbose=2)
print("Accuracy for LSTM Model: %.2f%%" % (scores[1]*100))
model.save('Flood Detection using Keras LSTM.h5')
model.save_weights('Flood Detection using Keras LSTM.h5')




print('Building CNN Model')
embedding_layer = Embedding(input_dim = max_features, output_dim = embedding_dim, input_length=max_length, weights = None, trainable=True)
sequence_input = Input(shape=(max_length,), dtype='float32')
embedded_sequences = embedding_layer(sequence_input)
l_cov1 = Convolution1D(filters =128, kernel_size = 3, strides=1, activation='relu', use_bias=False, border_mode='same')(embedded_sequences)
l_pool1 = MaxPooling1D(pool_size = 3)(l_cov1)
l_cov2 = Convolution1D(filters = 64, kernel_size = 5, strides=1, activation='relu', border_mode='same')(l_pool1)
l_pool2 = MaxPooling1D(pool_size = 5)(l_cov2) 
l_flat = Flatten()(l_pool2)
l_dense1 = Dense(output_dim = 64, activation='relu')(l_flat)
l_drop1 = Dropout(0.2)(l_dense1)
l_dense2 = Dense(output_dim = 32, activation='relu')(l_drop1)
l_drop2 = Dropout(0.2)(l_dense2)
preds = Dense(output_dim = 1, activation='sigmoid')(l_drop2)

model2 = Model(sequence_input, preds)
model2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model2.summary()
# model5.load_weights('Model4_Weights.h5', by_name=True)
model2.fit(train_vecs_w2v, y_train, epochs=35, batch_size=32, verbose=2)
scores= model2.evaluate(test_vecs_w2v, y_test, verbose=2)
print("Accuracy for CNN Model: %.2f%%" % (scores[1]*100))
model2.save('Earthquake Detection using Keras CNN.h5')
model2.save_weights('Earthquake Detection using Keras CNN.h5')

