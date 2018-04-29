import numpy as np
import csv
import sys
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from create_empty_db import Base, Comment
np.random.seed(1234)

from preprocess import make_sparse_array


# Globals
vocab_size = 7000
split = 0
max_features = 100
all_data = []
path = 'data/train.db'
engine = create_engine('sqlite:///'+path)
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()


def populate_array(data_list):
    # Take train or test list and split into data, label files.
    ret_array = np.ndarray(shape=(len(data_list), max_features, vocab_size))
    print(data_list[1])
    sentences = [sentence for (_,sentence),_ in data_list]
    for i in range(len(ret_array)):
        ret_array[i] = make_sparse_array(sentences[i])
    label_array = np.asarray([labels for _,labels in data_list])
    return ret_array, label_array


def main():
    with open("train.csv", 'r') as train_data:
        hold = csv.reader(train_data, delimiter=",", quotechar='"')
        for instance in hold:
            all_data.append(((instance[0], instance[1]),
                            [instance[2], instance[3], instance[4],
                            instance[5],instance[6], instance[7]]))
        # Remove labels
        del all_data[0]

    # Shuffle data
    np.random.shuffle(all_data)
    print('Adding data to database...')
    counter = 0
    for comment in all_data:
        comment_adder = None
        text_vector = make_sparse_array(comment[0][1])
        comment_adder = Comment(id=comment[0][0], text=comment[0][1],
                                vector=text_vector.dumps(),
                                toxic=comment[1][0],
                                severe_toxic=comment[1][0],
                                obscene=comment[1][0],
                                threat=comment[1][0],
                                insult=comment[1][0],
                                identity_hate=comment[1][0])
        session.add(comment_adder)
        session.commit()
        sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
        sys.stdout.write('\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
        sys.stdout.write('Comments added: %s' % str(counter))
        sys.stdout.flush()
        counter += 1
    print('\n\nComplete!')

    print('\n\nUse np.loads() when loading vectors')
'''    split = int(round(len(all_data) * 0.25, 0))
    train_data = all_data[split:]
    test_data = all_data[:split]
'''
if __name__ == '__main__':
    main()
