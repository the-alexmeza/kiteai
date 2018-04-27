'''
To improve accuracy, it is absolutely imperative that we eliminate as many
preprocessing errors as possible.

The tasks that need to be performed are:
    * URL removal
    * Email removal
    * Lowercasing
    * Nltk word_tokenize
    * Remove punctuation

In future or needing higher accuracy:
    * Stemming or Lemmatization

IMPORTANT: Each sparse matrix is NOT [word, position]. It is:
[# of sentence, (dictionary with words embedded)]
'''
import os
import string
import re

import numpy as np
import dill as pickle
import pickle as pkl

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import roc_auc_score
from collections import defaultdict

from sklearn.feature_extraction.text import CountVectorizer


# Create globals
table = str.maketrans({ch: " " for ch in string.punctuation})
lemma = WordNetLemmatizer()
vectorizer = pickle.load(open('vectorizer.p', 'rb'))
max_words = 100
feature_number = 7000
stop_words = set(stopwords.words('english'))

"""
if os.path.exists('vectorizer.kai'):
    vectorizer = pickle.load(open('vectorizer.kai', 'rb'))
else:
    print("Making vectorizer")
"""

# Tokenize sentence
def make_tokens(no_punct_sentence):
    return word_tokenize(no_punct_sentence)


# Remove punctuation
def remove_punctuation(sentence):
    # Removes URLs
    sentence = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '',
                      sentence, flags=re.MULTILINE)
    sentence = re.sub(r'\\n|\\t|\\s', ' ', sentence, flags=re.MULTILINE)
    sentence = re.sub(r'[0-9]', '', sentence, flags=re.MULTILINE)
    split_sentence = sentence.split(' ')
    no_punct = [s.translate(table) for s in split_sentence if s != '']
    joined_no_punct = ' '.join([word for word in no_punct])
    return joined_no_punct


# Pass through a lemmatizer
def lemmatized_tokens(tokens):
    word_stems = [lemma.lemmatize(token, pos='v') for token in tokens]
    rem_stopwords = [w for w in word_stems if not w in stop_words]
    return word_stems


# Combine all and run, returns list of stems
def preprocess(sentence):
    removed_punctuation = remove_punctuation(sentence.lower())
    sentence_to_tokens = make_tokens(removed_punctuation)
    lemma_token_list = lemmatized_tokens(sentence_to_tokens)
    most_common_vocab = pkl.load(open('vocab.p', 'rb'))
    mapping = defaultdict(lambda: 'UNK')
    for word in most_common_vocab:
        mapping[word] = word
    vocabulary_limit = [mapping[word] for word in lemma_token_list]
    s = ' '.join(vocabulary_limit)
    return s


# To make dictionary
def preprocess_for_dict(sentence):
    removed_punctuation = remove_punctuation(sentence.lower())
    sentence_to_tokens = make_tokens(removed_punctuation)
    lemma_token_list = lemmatized_tokens(sentence_to_tokens)
    return lemma_token_list


# Makes initial vectorizer model
def make_vectorizer(raw_docs):
    vocab = pkl.load(open('vocab.p', 'rb'))
    vectorizer = CountVectorizer(input='content',
                                 preprocessor=None,
                                 vocabulary=vocab,
                                 analyzer='word')
    count_train = vectorizer.fit(raw_docs)
    print(str(vectorizer.vocabulary_))
    print(str(len(vectorizer.vocabulary_)))
    with open('vectorizer.p', 'wb') as vectorizer_file:
        pickle.dump(count_train, vectorizer_file)
    print('Vectorizer created and saved.')


# Return vectorizer
def get_vectorizer():
    return vectorizer


# Makes 2D sparse array of individual words in dictionary.
def make_sparse_array(sentence):
    sparse_array_direct = vectorizer.transform([preprocess(sentence)])
    sparse_array = sparse_array_direct.toarray()
    sparse_array_length, sparse_array_height = sparse_array.shape
    full_matrix = np.zeros(shape=(max_words, feature_number))
    for i in range(max_words):
        if i < sparse_array_length and i < max_words:
            full_matrix[i] = sparse_array[i]
        else:
            pass
    return full_matrix


# Converts 2D sparse array to 1D
def convert_to_one_dimension(sparse_array):
    one_dimension_array = np.zeros(shape=(1, feature_number))
    for index in range(len(sparse_array)):
        if 1 in sparse_array[index]:
            one_dimension_array[0][sparse_array[index].argmax()] = 1
        else:
            pass
    return one_dimension_array


# Shows informative features
def show_most_informative_features(clf, n=50):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))


def get_auc(y_true, y_scores):
    print(str(roc_auc_score(y_true, y_scores)))

# Run on main
if __name__ == '__main__':
    while True:
        sentence = input('Sentence :: ')
        print(preprocess(sentence))
        print(make_sparse_array(preprocess(sentence)))
        print(make_sparse_array(preprocess(sentence)).shape)
