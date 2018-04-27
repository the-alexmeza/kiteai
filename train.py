import pickle
import random
import numpy as np
import csv

from preprocess import (make_sparse_array, convert_to_one_dimension,
                        show_most_informative_features, get_vectorizer)
from sklearn.naive_bayes import BernoulliNB


results = {}
combined_store = []
training_number = 5000
feature_number = 276


bayes = BernoulliNB()


with open('train.csv') as csvfile:
    data = csv.reader(csvfile)
    count = 0
    for row in data:
        if count > training_number:
            break
        else:
            label = 1 if row[2] == '1' else 0
            combined_store.append((convert_to_one_dimension(
                                   make_sparse_array(row[1])),
                                   label))
            count += 1


random.shuffle(combined_store)


multip = round(len(combined_store) * 0.33)

train_data = combined_store[multip:]
test_data = combined_store[:multip]
print('Split test/train')
X_train = np.reshape(np.array([text for (text, label) in train_data]),
                     (len(train_data), feature_number))
y_train = np.array([label for (text, label) in train_data])
X_test = np.reshape(np.array([text for (text, label) in test_data]),
                    (len(test_data), feature_number))
y_test = np.array([label for (text, label) in test_data])
print('Fitting model')
print('X_train shape ', X_train.shape)
print('X_test ', X_test.shape)
print('y_train ', y_train.shape)
print('y_test ', y_test.shape)
print(np.array_str(X_train))
print(np.array_str(X_test))
print(np.array_str(y_test))
print(np.array_str(y_train))
'''ri = [0, 1]
for i in range(len(X_train)):
    for item in X_train[i]:
        if item not in ri:
            print('X-train ', item)
for i in range(len(X_test)):
    for item in X_test[i]:
        if item not in ri:
            print('X-test ', item)
for i in range(len(y_train)):
    if y_train[i] not in ri:
        print('y-train ', y_train[i])
for i in range(len(y_test)):
    if y_test[i] not in ri:
        print('y-test ', y_test[i])'''
bayes.fit(X_train, y_train)
print('Finished fit')
pred = bayes.predict(X_test)
score = bayes.score(X_test, y_test)

print("Predictions: " + str(pred))
print("Score: ", score)

show_most_informative_features(bayes)


with open('model_gaussian.pkl', 'wb') as model_writer:
    pickle.dump(bayes, model_writer)
