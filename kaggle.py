import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import joblib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/home/svogeti/Documents/ATIS-Dataset/284285_585165_bundle_archive'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import spacy
import csv

global dictOfWords
def read_data(path):
    with open(path, 'r') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        labels = []
        sentences = []
        for row in readCSV:
            label = row[0]
            sentence = row[1]
            labels.append(label)
            sentences.append(sentence)
    return sentences, labels


sentences_test,labels_test = read_data('/home/svogeti/Documents/ATIS-Dataset/284285_585165_bundle_archive/atis_intents_test.csv')

sentences_test=['Quality Profile Access']
#labels_test=['atis_ground_service','atis_airfare','atis_flight']
print(sentences_test,'\n')
print(set(labels_test))
dictOfWords = { i : list(set(labels_test))[i] for i in range(0, len(list(set(labels_test))))}

# Loading Training Data

sentences_train,labels_train = read_data('/home/svogeti/Documents/ATIS-Dataset/284285_585165_bundle_archive/atis_intents_train.csv')

# Load the spacy model: nlp
nlp = spacy.load('en_vectors_web_lg')
embedding_dim = nlp.vocab.vectors_length

print(embedding_dim)

def encode_sentences(sentences):
    # Calculate number of sentences
    n_sentences = len(sentences)

    print('Length :-',n_sentences)

    X = np.zeros((n_sentences, embedding_dim))
    #y = np.zeros((n_sentences, embedding_dim))

    # Iterate over the sentences
    for idx, sentence in enumerate(sentences):
        # Pass each sentence to the nlp object to create a document
        doc = nlp(sentence)
        # Save the document's .vector attribute to the corresponding row in
        # X
        X[idx, :] = doc.vector
    return X
train_X = encode_sentences(sentences_train)
test_X = encode_sentences(sentences_test)

def label_encoding(labels):
    # Calculate the length of labels

    n_labels = len(labels)
    print('Number of labels :-',n_labels)


    # import labelencoder
    from sklearn.preprocessing import LabelEncoder
    # instantiate labelencoder object
    le = LabelEncoder()
    y =le.fit_transform(labels)
    print("Preprocessing classes")
    global actualList
    actualList=list(le.classes_);
    print(list(le.classes_))
    print(list(le.inverse_transform(y)))
    print(y[:100])
    print('Length of y :- ',y.shape)
    return y

train_y = label_encoding(labels_train)
test_y = label_encoding(labels_test)

df1 = pd.read_csv('/home/svogeti/Documents/ATIS-Dataset/284285_585165_bundle_archive/atis_intents_train.csv', delimiter=',')
df1.dataframeName = 'atis_intents_train.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')

df1.sample(10)
df1.describe()

import matplotlib.pyplot as plt
import seaborn as sns

# matplotlib histogram
plt.hist(train_y)

# Add labels
plt.title('Histogram of Intent Lables')
plt.xlabel('Intent Types')
plt.ylabel('Frequency')
#df1['atis_flight'].hist()

# Import SVC
from sklearn.svm import SVC
# X_train and y_train was given.
def svc_training(X,y):
    # Create a support vector classifier
    clf = SVC(C=1)

    # Fit the classifier using the training data
    clf.fit(X, y)
    return clf

#model = svc_training(train_X,train_y)

def svc_validation(model,X,y):
    # Predict the labels of the test set
    y_pred = model.predict(X)

    # Count the number of correct predictions
    n_correct = 0
    for i in range(len(y_pred)):
        print("Y[i]")
        print(y[i])
        print("y_pred[i]")
        print(actualList[y_pred[i].item()])
        #if y_pred[i] == y[i]:
        #    return actualList[y_pred[i].item()]
        #    n_correct += 1
        return actualList[y_pred.item()]
    print("Predicted {0} correctly out of {1} training examples".format(n_correct, len(y)))


#svc_validation(model,train_X,train_y)
#svc_validation(model,test_X,test_y)

#def getPickleFile():
#    import pickle

#    # Save the trained model as a pickle string.
    saved_model = pickle.dumps(model)

    # Load the pickled model
#    print("Achar Test")
#    knn_from_pickle = pickle.loads(saved_model)

    # Use the loaded pickled model to make predictions
#    print("Achar Test1")
    # save the model to disk
#    global filename_model
 #   filename_model='finalized_model.sav'
 #   joblib.dump(knn_from_pickle, 'finalized_model.sav')

def getPredictions(modelFile,testInput):
    # load the model from disk
    loaded_model = joblib.load(modelFile)
    result=svc_validation(loaded_model, testInput, test_y)
    return result


