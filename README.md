# Twitter-Event-Detection
This repository contains analysis of different machine learning and deep learning techniques. It is a comparison based study of  different techniques such as Convolutional Neural Network, LSTM, Perceptron, Stochastic Gradient Descent, Logistic Regression, Bernoulli Naive Bayes, Gaussian Naive Bayes etc.
In this work I have used Keras with Theano Backend for the Convolutional Neural Network and LSTM RNN and Scikit Learn for other different classifiers like Logistic Regression, SDG, BernoulliNB.

Besides event detection, these learning algorithms can be used for sentiment analysis and other multiclass classification.

I have used GloVe word vectors to convert each tweet into a tweet vector of vector length 300. Used TFIDF(Term Frequency Inverse Document Frequency) to punish high frequency words in tweets.
Here is a link to GloVe: https://nlp.stanford.edu/projects/glove/

For dataset, I have used different datasets, one of the most recent ones I have used is shared by InfoLab-USC, https://github.com/sifatron/bdr-tweet, they have done a very good study on Twitter Event Detection for natural disasters.
