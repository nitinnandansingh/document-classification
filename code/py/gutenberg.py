import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import io
import nltk
#nltk.download('stopwords')
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

dataset = pd.read_csv(r'/Users/sidraaziz/PycharmProjects/ProjectGutenberg/master996.csv', sep=';', engine='python')

row_count = len(dataset.axes[0])
data_list = []  # list containing all text data
data_class = []  # class of the respective text data retrieved

for i in range(0, row_count):
    bookid = dataset.iloc[i, 1]
    bookid_split = bookid.split('.')
    data_class.append(dataset.iloc[i, 2])
    url = 'file:///' + '/Users/sidraaziz/PycharmProjects/ProjectGutenberg/Gutenberg_19th_century_English_Fiction/' + bookid_split[0] + '-content.html'
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
#obj_dataset['Data_Content'] = obj_dataset['Data_Content'].str.lower()
example_lower = obj_dataset.iloc[0]
#print("this is lowercase implementation example:")
#print(example_lower)



#tokenizing the dataset row-wise:
row_count = len(dataset.axes[0])
token_list = []
obj_dataset['Data_Content'] = obj_dataset['Data_Content'].astype(str)

for i in range(0, row_count):  ###specify the range
    tokenized = word_tokenize(obj_dataset.loc[i, "Data_Content"])
    token_list.append(tokenized)

# print(token_list[0])

# Adding token_list into existing dataset:
dictionary = {'Data_Content': data_list, 'Target_Class': data_class, 'Tokenized_Data_Content': token_list}
obj_dataset = pd.DataFrame(dictionary, columns=['Data_Content', 'Target_Class', 'Tokenized_Data_Content'])
# print(obj_dataset.head())

# tokenizing the dataset row-wise:
row_count = len(dataset.axes[0])
token_list = []
obj_dataset['Data_Content'] = obj_dataset['Data_Content'].astype(str)

for i in range(0, row_count):  ###specify the range
    tokenized = word_tokenize(obj_dataset.loc[i, "Data_Content"])
    token_list.append(tokenized)

# print(token_list[0])

# Adding token_list into existing dataset:
dictionary = {'Data_Content': data_list, 'Target_Class': data_class, 'Tokenized_Data_Content': token_list}
obj_dataset = pd.DataFrame(dictionary, columns=['Data_Content', 'Target_Class', 'Tokenized_Data_Content'])
# print(obj_dataset.head())


# tokenizing the dataset row-wise:
row_count = len(dataset.axes[0])
token_list = []
obj_dataset['Data_Content'] = obj_dataset['Data_Content'].astype(str)

for i in range(0, row_count):  ###specify the range
    tokenized = word_tokenize(obj_dataset.loc[i, "Data_Content"])
    token_list.append(tokenized)

# print(token_list[0])

# Adding token_list into existing dataset:
dictionary = {'Data_Content': data_list, 'Target_Class': data_class, 'Tokenized_Data_Content': token_list}
obj_dataset = pd.DataFrame(dictionary, columns=['Data_Content', 'Target_Class', 'Tokenized_Data_Content'])
# print(obj_dataset.head())


import nltk
#nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

stemming = PorterStemmer()
stop = nltk.corpus.stopwords.words('english')
# stop = set(stopwords.words("english"))
newStopWords = ['b', 'p', 'would', 'could']
stop.extend(newStopWords)


def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X

def clean_text(raw_text):
    # """"This function works on a raw text string, and:
    # 1) changes to lower case
    # 2) tokenizes (breaks down into words
    # 3) removes punctuation and non-word text
    # 4) finds word stems
    # 5) removes stop words
    # 6) rejoins meaningful stem words"""

    # Convert to lower case
    text = raw_text.lower()

    # Tokenize
    tokens = nltk.word_tokenize(text)

    # Keep only words (removes punctuation + numbers)
    # use .isalnum to k eep also numbers
    token_words = [w for w in tokens if w.isalpha()]

    # Stemming
    stemmed_words = [stemming.stem(w) for w in token_words]

    # Remove stop words
    meaningful_words = [w for w in stemmed_words if not w in stop]

    # Rejoin meaningful stemmed words
    joined_words = (" ".join(meaningful_words))

    # Return cleaned data
    return joined_words


### APPLY FUNCTIONS TO EXAMPLE DATA
obj_dataset['Data_Content'] = obj_dataset['Data_Content'].astype(str)
# Get Content to clean
content_to_clean = list(obj_dataset['Data_Content'])

# Clean content
cleaned_content = apply_cleaning_function_to_list(content_to_clean)

# Show first example
# print ('Original text:',content_to_clean[0])
# print ('\nCleaned text:', cleaned_content[0])

# Add cleaned data back into DataFrame
obj_dataset['cleaned_Data_Content'] = cleaned_content


#stopword removal on tokenized data set
stop = nltk.corpus.stopwords.words('english')
from nltk.stem import PorterStemmer
stem = PorterStemmer()

newStopWords = ['b','p']
stop.extend(newStopWords)

obj_dataset['Tokenized_Data_Content'] = obj_dataset['Tokenized_Data_Content'].apply(lambda x: [item for item in x if item not in stop])
# Stemming on tokenized data set
#obj_dataset['Tokenized_Data_Content'] = obj_dataset['Tokenized_Data_Content'].apply(lambda x: [item for item in x if item not in stem])
print(obj_dataset.head())

frequency_dist = obj_dataset['cleaned_Data_Content'].value_counts()[:5].sort_values(ascending=False)
print(frequency_dist)
# create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(obj_dataset, y, test_size=0.25)
#print('Data for X_train before stopword removal:')
#print(X_train.shape, y_train.shape)
print()
#print(X_test.shape, y_test.shape)

