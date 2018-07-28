import os
import collections
import sys
import keras
import math
import numpy as np
import tensorflow as tf
import reader
from config import m_config
from model import LanguageModel
from keras.utils import to_categorical

def main():
    DATA_PATH = os.getcwd()+"/simple-examples/data/"
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)
    x_train,y_train= reader.cut_num_step(train_data[0:10001],m_config.nb_time_steps)
    features = x_train.astype('float32')
    labels = y_train.astype('float32')
    dataset_x = tf.data.Dataset.from_tensor_slices(features).repeat()
    dataset_y = tf.data.Dataset.from_tensor_slices(labels).repeat()
    dataset_x = dataset_x.batch(m_config.batch_size)
    dataset_y = dataset_y.batch(m_config.batch_size)
    itera_x = dataset_x.make_one_shot_iterator()
    itera_y = dataset_y.make_one_shot_iterator()
    config = m_config
    train_model = LanguageModel()
    print(train_model.build_lm_model(itera_x,itera_y))


if __name__ == "__main__":
    main()


