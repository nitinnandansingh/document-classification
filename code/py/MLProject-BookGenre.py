import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import io
import nltk
nltk.download('stopwords')
import re
plt.show()
from sklearn.feature_extraction.text import *
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from sklearn.preprocessing import *
from numpy import argmax
import warnings
warnings.filterwarnings("ignore")

import os
import nltk
import urllib
from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk import word_tokenize
import codecs
import pandas as pd

cwd = os.getcwd()
#testing with codecs to open a html file, not necessarily needed though.
import codecs
data1 = codecs.open('/Users/sidraaziz/PycharmProjects/MLProject/Gutenberg_19th_century_English_Fiction/pg11CarolAlice-content.html','r')
#print(data1.read())

dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/DKEProject/venv/master996.csv', sep=';', engine='python')

row_count = len(dataset.axes[0])
data_list = []  # list containing all text data
data_class = []  # class of the respective text data retrieved

for i in range(0, row_count):
    bookid = dataset.iloc[i, 1]
    bookid_split = bookid.split('.')
    data_class.append(dataset.iloc[i, 2])
    url = 'file:///' + '/Users/sidraaziz/PycharmProjects/MLProject/Gutenberg_19th_century_English_Fiction/' + bookid_split[0] + '-content.html'
    data_list.append(urllib.request.urlopen(url).read())
   # print(data_list[0])

#adding the data_list and data_class to dataframe
dictionary = {'Data_Content':data_list,'Target_Class': data_class}
dataset = pd.DataFrame(dictionary, columns=['Data_Content','Target_Class'])
#print(dataset.head())

#Now dataframe created above can be split into Features (X) and Target class (y)
le_target = LabelEncoder()
obj_dataset= dataset.select_dtypes(include=['object']).copy()
obj_dataset['genreLabel'] = le_target.fit_transform(obj_dataset['Target_Class'])
print()
#print('below is the list seperation with Target_Class encoded:')
#print(obj_dataset[['genreLabel', 'Target_Class']])
features = ['Data_Content']
# Separating out the features
X = obj_dataset.loc[:, features].values
# Separating out the target
y = obj_dataset.loc[:,['Target_Class']].values

#Next Steps: Work on feature extraction, train and test split of the data.

#X = StandardScaler().fit_transform(X)
#print('After data standardardisation:')
#print(pd.DataFrame(data = X, columns = features).head())
print()
#print('genre distribution')
#print(obj_dataset['genreLabel'].value_counts())

#onehotencoding on above label encoder
onehot_encoder = OneHotEncoder(sparse=False)
obj_dataset['genreLabel'] = obj_dataset['genreLabel'].values.reshape(len(obj_dataset['genreLabel']), 1)
onehot_encoded = onehot_encoder.fit_transform(obj_dataset['genreLabel'].values.reshape(-1,1))
#print(onehot_encoded)

# invert first example
inverted = le_target.inverse_transform([argmax(onehot_encoded)])
#print (onehot_encoded[970])

#removing special characters, extra whitespaces, digits, stopwords and lower casing the text corpus
wordpunc_tokenize = nltk.WordPunctTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
#converting data into lower case:
obj_dataset['Data_Content'] = obj_dataset['Data_Content'].str.lower()
example_lower = obj_dataset.iloc[0]
#print("this is lowercase implementation example:")
#print(example_lower)


# create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(obj_dataset, y, test_size=0.25)
#print('Data for X_train before stopword removal:')
#print(X_train.shape, y_train.shape)
print()
#print(X_test.shape, y_test.shape)

# -- stopwords and tf-idf implementation --
#Removing stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(lowercase=True,stop_words='english')
X_train_counts = count_vect.fit_transform(X_train['Data_Content'])
#print('Data for X_train_counts after stopword removal:')
#print(X_train_counts.shape)

print('Original Content: %s' % (obj_dataset['Data_Content']))
print()
obj_dataset['tokenized_Data_Content'] = obj_dataset['Data_Content'].apply(tokenizer.tokenize)

#print(nltk.word_tokenize(str(obj_dataset['Data_Content'])))

#TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print('TF-IDF transformation:')
print(X_train_tfidf.shape)

#filtered_sent=[]
#for w in tokenized_sent:
  #  if w not in stop_words:
 #       filtered_sent.append(w)
#print("Tokenized Sentence:",tokenized_sent)
#print("Filterd Sentence:",filtered_sent)
