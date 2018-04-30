import keras
from keras.layers import Input, Dense, Activation, Dropout
import pickle as pkl
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from create_empty_db import Base, Comment

path = 'data/train.db'
engine = create_engine('sqlite:///'+path)
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

last_id = 0


def LoadTrainBatch(batch_size):
    for item in session.query(Comment).\
            filter(item('id>last_id and id<='+last_id+batch_size)).all():
        print(text.id)
    return samples, labels

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
