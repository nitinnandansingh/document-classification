# -*- coding: utf-8 -*-
"""TopicModeling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jv4wWbylYibTTdXkwDe66tcWG-DiVd3E

# Topic Modeling
This program tries classify books according to their genre based on topic modeling using LDA. To do so, first a number of topics gets exxtracted from the books in the test set. Using with which probability a topic is present in a certain book and which genre it is, we can train a classifier. This classifier can then be used to determine for books from the test set, which genre they might have.

# Program
"""

from google.colab import drive
from sklearn.model_selection import StratifiedShuffleSplit
from gensim import corpora
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from os import listdir
from os.path import isfile, join
import pandas as pd
import gensim
import numpy as np
import random
from google.colab import output
import nltk
from gensim.models import Phrases
nltk.download('wordnet')
nltk.download('names')
from nltk.corpus import names
from nltk.corpus import wordnet as wn

NUM_TOPICS = 5                # Number of topics to generate. Equals size of generated data
USE_FIRST_N_WORDS = 2500      # Number of tokens to use for the topic generation
SKIP_RANDOM_LITERARAY = False # For testing purposes ignore some literary books
SKIP_PERCANTAGE = 0.8         # 80 Percent of all literary books will be skipped (randomly)
SAVE_PREPARED = True          # Saves the completely processed tokens as a np array
LOAD_PREPARED = False         # Load in the completely processed tokens to save time

"""## Load in the already tokenized books

You need to load in both csv files, the master file and the lemmatized content.
"""

drive.mount('/content/drive')

pathLemmat = "/content/drive/My Drive/ATIML/lemmatized_content.csv"
pathMaster = "/content/drive/My Drive/ATIML/master996.csv"

lemmat = pd.read_csv( pathLemmat )
master = pd.read_csv( pathMaster, encoding='windows-1252' , sep=';' )

books = []
genres = []

count = 1

for index, row in lemmat.iterrows():

  genre = master['guten_genre'][ index ]

  # Skip some literary books to improve the runtime for testing purposes
  if SKIP_RANDOM_LITERARAY:
    if genre == 'Literary':
        val = random.random()
        if val > 1 - SKIP_PERCANTAGE:
          continue

  # Get the tokens and remove words that are too short
  tokens = []
  try:
    tokens = row['cleaned_Data_Content'].split(' ')
    tokens = [ token for token in tokens if len(token) > 3 ]
  except:
    print('A book failed')

  books.append( tokens )
  genres.append( genre )

  output.clear('status_text')
  with output.use_tags('status_text'):
    print( 'Books loaded: ' + str( count ))
  count += 1

genres = np.array( genres )
books = np.array( books )

# Additional data processing to optimize the topics generated

commonWords = []

with open( "/content/drive/My Drive/ATIML/common.txt" ) as common:
  for word in common:
      commonWords.append( word[:-1] )

forbidden = [
             'littl',
             'think',
             'befor',
             'thought',
             'never',
             'thing',
             'shall',
             'someth',
             'every',
             'found',
             'seemed',
             'looked',
             'turned',
             'called'
]

count = 1

tmp = []
for b in books:

  # Use only a subset of tokens
  m = min( len(b), USE_FIRST_N_WORDS)

  l = b[:m]
  l = [ t for t in l if (t not in names.words('male.txt'))]
  l = [ t for t in l if (t not in names.words('female.txt'))]
  l = [ t for t in l if (t not in forbidden) ]
  l = [ t for t in l if (t not in commonWords)]
 
  tmp.append( l )

  output.clear('status_text')
  with output.use_tags('status_text'):
    print( 'Books loaded: ' + str( count ))
  count += 1

books = tmp
books = np.array( books )

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(books, min_count=20)
for idx in range(len(books)):
    for token in bigram[books[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            books[idx].append(token)

"""Save and load funcitonality of the completely processed tokens to speed up the whole process in future runs."""

data_path = 'prepared_tokens.npy'

if SAVE_PREPARED:
  with open(data_path, 'wb') as f:
    np.save(f, genres)
    np.save(f, books)

if LOAD_PREPARED:
  with open(data_path, 'rb') as f:
    genres = np.load(f)
    books = np.load(f)

"""## Create a train test split"""

NUMBER_OF_SPLITS = 5
TEST_SIZE = 1 / 3

sss = StratifiedShuffleSplit(
    n_splits=NUMBER_OF_SPLITS,
    test_size=TEST_SIZE,
    random_state=0
)

splits = sss.split( books, genres )

"""## Get the Topics and convert them to usable data"""

# TODO add your train test split here, make sure to fill in 

train_index = []
test_index = []

for tr, te in splits:
  train_index = tr
  test_index = te
  
books_train, books_test = books[train_index], books[test_index]
genres_train, genres_test = genres[train_index], genres[test_index]

# Create the topics
NUM_WORDS  = 6

dictionary = corpora.Dictionary( books_train )
dictionary.filter_extremes(no_below=20, no_above=0.5)
corpus = [ dictionary.doc2bow( text ) for text in books_train ]

# Set training parameters.
num_topics = NUM_TOPICS
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

ldamodel = gensim.models.ldamodel.LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

# Print out the topics
topics = ldamodel.print_topics( num_words=NUM_WORDS )
print("The following topics were generated:")
for topic in topics:
  print( topic )

# Process the books and get final training data
X_train = []
Y_train = genres_train # TODO do I have to preprocess it as well?

for book in books_train:

  # Get the topic weights
  bow = dictionary.doc2bow( book )
  topics = ldamodel.get_document_topics( bow )

  # Convert the vector of dynamic length to
  # constant length feature vector
  x = [0] * NUM_TOPICS
  for topic in topics:
    x[topic[0]] = topic[1]
  X_train.append(x)

# Prepare our test data in the same way
X_test = []
Y_test = genres_test # TODO do I have to preprocess it as well?

for book in books_test:

  # Get the topic weights
  bow = dictionary.doc2bow( book )
  topics = ldamodel.get_document_topics(bow)

  # Convert the vector of dynamic length to
  # constant length feature vector
  x = [0] * NUM_TOPICS
  for topic in topics:
    x[topic[0]] = topic[1]
  X_test.append(x)