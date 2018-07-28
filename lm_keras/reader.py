
import sys
import os
import collections
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
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


def cut_num_step(train_data, num_step):
    x = []
    y = []
    a = []
    b = []
    for i in range(len(train_data)):
        if i % num_step == 0 and i != 0:
            x.append(a)
            y.append(b)
            a = []
            b = []
        if i == len(train_data) - 1:
            b.append(0)
        else:
            a.append(train_data[i])
            b.append(to_categorical(train_data[i + 1], num_classes=10000))
    return np.asarray(x), np.asarray(y)
