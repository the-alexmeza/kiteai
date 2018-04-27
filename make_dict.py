import csv
import nltk
import numpy as np
import pickle as pkl

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from preprocess import preprocess_for_dict


vocab_size = 7000
stop_words = set(stopwords.words('english'))
# Format of list:
# [((ID, Text), [toxic, severe_toxic, obscene, threat, insult, identity_hate])]
all_data = []
all_tokens = []

with open("train.csv", 'r') as train_data:
    hold = csv.reader(train_data, delimiter=",", quotechar='"')
    for instance in hold:
        all_data.append(((instance[0], instance[1]),
                        np.array([instance[2], instance[3], instance[4],
                                  instance[5],instance[6], instance[7]])))
    # Remove labels
    del all_data[0]

# Retrieve all sentences
all_text = [sentence for ((_, sentence), _) in all_data]
for sentence in all_text:
    t = preprocess_for_dict(sentence)
    for item in t:
        if item not in stop_words:
            all_tokens.append(item)
vocab = nltk.FreqDist(all_tokens)
most_common_vocab = [word for word,_ in vocab.most_common(vocab_size)]
print(most_common_vocab)
pkl.dump(most_common_vocab, open('vocab.p', 'wb'))
pkl.dump(all_text, open('corpus.p', 'wb'))
