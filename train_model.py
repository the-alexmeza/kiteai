import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input, LSTM
import pickle as pkl
from preprocess import make_sparse_array

train_data = pkl.load(open('data/train_samples.p', 'rb'))
train_labels = pkl.load(open('data/train_labels.p', 'rb'))
test_data = pkl.load(open('data/test_samples.p', 'rb'))
test_labels = pkl.load(open('data/test_labels.p', 'rb'))

model = Sequential()

model.add(Dense(67, input_dim=15))
model.add(Dense(1))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# one_hot_labels = keras.utils.to_categorical(train_labels)

model.fit(train_data, train_labels, epochs=10)
print()
score = model.evaluate(test_data, test_labels)

prediction = model.predict(make_sparse_array('kablamo'))
print("Pred: ", prediction)
