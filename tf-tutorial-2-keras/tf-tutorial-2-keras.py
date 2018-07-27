import os
import collections
import sys
import keras
import math
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,Input
from keras.layers import Embedding,LSTM
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input, Dense,InputLayer

Py3 = sys.version_info[0] == 3
def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        if Py3:
            return f.read().replace("\n", "<eos>").split()
        else:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    word_to_id = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary

def cut_num_step(train_data,num_step):
    x = []
    y = []
    a = []
    b = []
    for i in range(len(train_data)):
        if i % num_step == 0 and i!=0:
            x.append(a)
            y.append(b)
            a =[]
            b =[]
        if i == len(train_data)-1:
            b.append(0)
        else:
            a.append(train_data[i])
            b.append(to_categorical(train_data[i+1],num_classes=10000))
    return np.asarray(x),np.asarray(y)


DATA_PATH = os.getcwd()+"/simple-examples/data/"
train_data, valid_data, test_data, _ = ptb_raw_data(DATA_PATH)
nb_time_steps = 20
x_train,y_train= cut_num_step(train_data[0:10001],nb_time_steps)
x_valid,y_valid= cut_num_step(valid_data,nb_time_steps)

MAX_SEQUENCE_LENGTH = 20
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
batch_size = 20
units = 200
nb_words = 9999
nb_time_steps = 20
nb_input_vector = 1
sgd_lr = 1.0
sgd_momentum = 0.9
sgd_decay = 0.0
nb_epoch = 3

features = x_train.astype('float32')
labels = y_train.astype('float32')
dataset_x = tf.data.Dataset.from_tensor_slices(features).repeat()
dataset_y = tf.data.Dataset.from_tensor_slices(labels).repeat()
dataset_x = dataset_x.batch(batch_size)
dataset_y = dataset_y.batch(batch_size)
itera_x = dataset_x.make_one_shot_iterator()
itera_y = dataset_y.make_one_shot_iterator()


x_valid,y_valid= cut_num_step(valid_data,nb_time_steps)
validation_data=(x_valid,y_valid)


inputs = Input(tensor=itera_x.get_next())
embedding_layer = Embedding(nb_words+1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)
X =  embedding_layer(inputs)
X = keras.layers.LSTM(units,
    activation='tanh', 
    recurrent_activation='hard_sigmoid', 
    use_bias=True, 
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',return_sequences=True)(X)
X = keras.layers.LSTM(units,
    activation='tanh', 
    recurrent_activation='hard_sigmoid', 
    use_bias=True, 
    kernel_initializer='glorot_uniform',
    recurrent_initializer='orthogonal',
    bias_initializer='zeros',return_sequences=True)(X)
predictions= Dense(nb_words+1,activation='softmax')(X)
sgd = SGD(lr=sgd_lr, momentum=sgd_momentum, decay=sgd_decay, nesterov=False)
model = Model(inputs=inputs, outputs=predictions)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'],target_tensors=[itera_y.get_next()])
print(model.summary())
history = model.fit(epochs=nb_epoch,steps_per_epoch=x_train.shape[0]//batch_size) 
print("PPL:",np.exp(np.array(history.history["loss"])))

