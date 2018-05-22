import pandas as pd

import re

import numpy as np

import nltk

import csv

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem.porter import PorterStemmer

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, accuracy_score, make_scorer

from sklearn import svm, cross_validation

from sklearn.cross_validation import train_test_split

#from unidecode import unidecode

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

names = ['Id', 'Text', 'Aspect Term', 'Term Location', 'Class']

names_1 = ['Id', 'Text', 'Aspect Term', 'Term Location']

data1 = pd.read_csv(r'C:\Users\USER\PycharmProjects\dara 1_train.csv', skiprows=1, names=names)

data2 = pd.read_csv(r'C:\Users\USER\PycharmProjects\Data-1_test.csv', skiprows=1, names=names_1)

data = pd.DataFrame(data1)

data_1 = pd.DataFrame(data2)

stop = set(stopwords.words('english'))

snowball = SnowballStemmer('english')


def preprocess(text):
    text = str(text)

    text = text.lower()

    text = text.replace('[comma]', '')

    text = re.sub(r'\d', '', text)

    text = re.sub(r'[^\w\s\']', '', text)

    text = text.replace("'", '')

    text = text.replace("(", '')

    text = text.replace(")", '')

    text = text.replace(",", '')

    text = text.replace('"', '')

    text = text.replace('.', '')

    text = text.replace('!', '')

    text = text.replace('?', '')

    text = text.replace('_', '')

    text = text.replace('/', ' ')

    text = text.replace(':', '')

    text = text.replace('-', ' ')

    text = text.replace('*', '')

    text = text.replace('=', '')

    text = text.replace(';', '')

    text = text.replace("\[]", "")

    text = text.replace("\_", "")

    text = text.replace("\d", "")

    tokenizer = RegexpTokenizer(r'\w+')

    text = text.replace('freesecuritysoftware', 'free security software')

    tokens = tokenizer.tokenize(text)

    filtered_Words = [w for w in tokens if not w in stop]

    stemmed_words = [snowball.stem(w) for w in filtered_Words]

    text = ' '.join(filtered_Words)

    return text


y_train = data['Class']

features = ['Text', 'Class']

# X_t, X_test, y_train, y_test = train_test_split(data[features], y, test_size=0.2)


X_test = []

X_train = []

for i in range(len(data['Text'])):
    X_train.append("".join(preprocess(data['Text'][i])))

#print(X_train)

for i in range(len(data_1['Text'])):
    X_test.append("".join(preprocess(data_1['Text'][i])))

#print(X_test)

vectorizer = TfidfVectorizer(min_df=0.00125,

                             max_df=0.70,

                             use_idf=True,

                             ngram_range=(1, 5))

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

clf = svm.SVC(kernel='linear')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

#y_test = data_1['Class']

#print(classification_report(y_test, y_pred))

#print(accuracy_score(y_test, y_pred))

f = open(r'C:\Users\USER\PycharmProjects\output.txt','w+')
for i in range(len(data_1['Id'])):
    print(data_1['Id'][i], ';;', y_pred[i], sep='', file=f)
    print(data_1['Id'][i], ';;', y_pred[i])

f.close()

# nested_score = cross_val_score(clf, X_dtm, y, cv=10, scoring=make_scorer(classification_report_with_accuracy_score))