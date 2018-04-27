import numpy as np
import pickle as pkl
import csv
np.random.seed(1234)

from preprocess import make_sparse_array

# Globals
vocab_size = 7000
split = 0
max_features = 100
all_data = []


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
    train_samples_file = open('train_samples.p', 'wb')
    train_labels_file = open('train_labels.p', 'wb')
    test_samples_file = open('test_samples.p', 'wb')
    test_labels_file = open('test_labels.p', 'wb')
    with open("train.csv", 'r') as train_data:
        hold = csv.reader(train_data, delimiter=",", quotechar='"')
        for instance in hold:
            all_data.append(((instance[0], instance[1]),
                            np.array([instance[2], instance[3], instance[4],
                                      instance[5],instance[6], instance[7]])))
        # Remove labels
        del all_data[0]

    # Shuffle data
    np.random.shuffle(all_data)
    split = int(round(len(all_data) * 0.25, 0))
    print(all_data[1])
    train_data = all_data[split:]
    test_data = all_data[:split]
    train_features, train_labels = populate_array(train_data)
    test_features, test_labels = populate_array(test_data)
    pkl.dump(train_features, train_samples_file)
    pkl.dump(train_labels, train_labels_file)
    pkl.dump(test_features, test_samples_file)
    pkl.dump(test_labels, test_labels_file)

if __name__ == '__main__':
    main()
