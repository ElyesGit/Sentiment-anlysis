import os
import time
import collections
import numpy as np
import pickle
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import sentiment
import spacy
import gensim
from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from multiprocessing import cpu_count
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from contextlib import suppress
# the following fixes some issue with the POS tagger
nltk.download('averaged_perceptron_tagger')

# Reading data 
def read_data(f_name):   
    curr_dir = os.path.dirname(os.path.realpath('tp2_NLP.py'))
    file_name = os.path.join(curr_dir,'downloads','data',f_name) 
    reviews=[]   
    ratings=[]
    sentences=[]
    # extracting the reviews
    with open(file_name, errors='ignore') as f:
        lines = f.readlines()
        for i in range(1,len(lines)):
            sentences.append(lines[i][3:len(lines[i])-4])
            reviews.append(str.lower(lines[i][3:len(lines[i])-4]).split())
            ratings.append(lines[i][len(lines[i])-2])
        f.close()
    return sentences, reviews, ratings

# Tokenization
def tokenization(sentences):
    new_revs=[]
    for rev in sentences:
        new_rev=[]
        ln= word_tokenize(rev)
        neg_marked = nltk.sentiment.util.mark_negation(ln)
        pos= nltk.pos_tag(ln)
        for elem in pos:
            # Extracting verbs, adverbs and adjectives
            if str(elem[1])[0:2] in ['JJ','RB','VB']:
                new_rev.append(elem[0])
        new_revs.append(new_rev)
    return new_revs

# Getting the embeddings for each selected token
def embedding(new_revs):
    reviews_in_emb_space= []
    for r in new_revs:
        emb= np.zeros(100)
        n=0
        for w in r:
            with suppress(Exception):
                emb+= model[str.lower(w)]
        reviews_in_emb_space.append(emb)
    return reviews_in_emb_space

# Evaluating the error rate
def error_rate(targets, ground_truth, exact=True):
    err=0
    for k in range(len(ground_truth)):
        with suppress(Exception):
            if exact== False and round(int(targets[k]))-int(ground_truth[k])>1:
                err+=1
            elif exact== True and round(int(targets[k])) != int(ground_truth[k]):
                err+=1
    return err

if __name__ == "__main__":

    # Reading the training data
    f_name= 'sentiment_dataset_train.csv'
    sentences, reviews, ratings= read_data(f_name)
    
    ##########   Training the classifier ###################
    # Obtaining the word embeddings from skip gram Word2Vec
    model= Word2Vec(reviews, min_count=5, workers= cpu_count(), sg=1) 
    
    # Extracting the embedding for the selected (sentiment) words
    new_revs= tokenization(sentences)
    reviews_in_emb_space= embedding(new_revs)
    
    ###########  Training the SVM classifier on the obtained embeddings
    clf = svm.SVC(kernel='linear') # Linear Kernel
    #clf.fit(reviews_in_emb_space, ratings)
    gnb = GaussianNB() # Naive Bayes
    gnb.fit(reviews_in_emb_space, ratings)
    
    ###########   To save the SVM classifier
    #with open('SVM_linear.pkl', 'wb') as f:
    #    pickle.dump(clf, f)
    #    f.close()
    
    ###########  To load the saved SVM model
    with open('SVM_linear.pkl', 'rb') as f:
        clf = pickle.load(f)
        f.close()
    
    ###########    Testing our classifier    ##########
    # Loading the test data
    f_name= 'sentiment_dataset_dev.csv'
    test_sentences, test_reviews, test_ratings= read_data(f_name)
    test_new_revs= tokenization(test_sentences)
    test_reviews_in_emb_space= embedding(test_new_revs)
    # Counting mis-classifications
    SVM_targets= clf.predict(test_reviews_in_emb_space)
    NB_targets= gnb.predict(test_reviews_in_emb_space)
    SVM_err=0
    NB_err=0
    for k in range(len(test_ratings)):
        with suppress(Exception):
            if round(int(SVM_targets[k])) != int(test_ratings[k]):
                SVM_err+=1
            if round(int(NB_targets[k])) != int(test_ratings[k]):
                NB_err+=1
    
    # Predicting reviews in the unlabeled test set
    f_name= 'sentiment_dataset_test.csv'
    curr_dir = os.path.dirname(os.path.realpath('tp2_NLP.py'))
    file_name = os.path.join(curr_dir,'downloads','data',f_name) 

    test_sentences, test_reviews, test_ratings= read_data(f_name)
    test_new_revs= tokenization(test_sentences)
    test_reviews_in_emb_space= embedding(test_new_revs)
    SVM_targets= clf.predict(test_reviews_in_emb_space)
    # Reading lies
    l=''
    for it in range(len(test_sentences)):
        l+= str(it)+ ',"'+ test_sentences[it]+ '",'+ str(SVM_targets[it])+ '\n'

    # Adding the predicted rating
    with open('testSVM_linear.txt', 'a',errors="ignore") as the_file:
        the_file.write(l)
        the_file.close()        
