# Sentiment-anlysis
A sentiment analysis of hotel reviews

In this project we explore some datasets containing hotel reviews and build two models predicting their ratings by extracting relevant information about them.

We start by tokenizing the corpus, extracting subjective words by marking negations and part of speech tags (POS). Secondly, we train a skip gram model to represent each word-token by a vector, hopefully capturing its semantics and hence the sentiment it was meant to convey.

Then, we train two models: a Naive Bayes classifier as a baseline model, and a linear kernel Support vector machine (SVM) as an upgrade, to classify the embedded word-vectors into their proper categories (i.e. ratings) on the training set, and evaluate them on the "dev" set. 

Finally, using the (already trained) SVM classifier, we attempt to predict the ratings of the reviews in the "test" set. 
