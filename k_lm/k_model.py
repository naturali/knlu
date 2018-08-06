import keras
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Bidirectional
from keras.layers import Embedding, LSTM
from keras.optimizers import SGD
from keras import backend as K



class Kmodel(keras.Model):
    def __init__(self, config):
        super(Kmodel, self).__init__()
        self.config = config
        self.embedding = Embedding(self.config.vocab_size,
                                   self.config.embedding_size,
                                   input_length=self.config.num_steps - 1)
        self.bi_lstm_1 = Bidirectional(
            LSTM(
                self.config.hidden_size,
                activation='tanh',
                recurrent_activation='hard_sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                return_sequences=True))
        self.dense_1 = Dense(self.config.vocab_size, activation='softmax')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.bi_lstm_1(x)
        x = self.dense_1(x)
        return x


class LanguageModel(object):
    def __init__(self, config):
        self.config = config


    def build_lm_model(self):
        self.model = Kmodel(self.config)
        # if self.config.gpu_num > 1:
        #     self.model = multi_gpu_model(self.model,gpus=self.config.gpu_num)
        # can't use multi_gpu becasue of the same reason of save function
        sgd = SGD(
            lr=self.config.learning_rate,
            momentum=0.1,
            decay=0.1,
            nesterov=False)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

    def fit_lm_model(self, inputs):
        iter_data = []
        for i in range(len(inputs)):
            inputs[i].init_reader(K.get_session())
            iter_data.append(inputs[i].epoch_input())
            K.get_session().run(iter_data[i].initializer)
        history = self.model.fit(x=tf.cast(iter_data[0].get_next()[0], dtype=tf.float32),
                                 y=tf.one_hot(iter_data[1].get_next()[3], self.config.vocab_size),
                                 epochs=self.config.max_epoch,
                                 steps_per_epoch=self.config.steps_per_epoch)
        ppl = np.exp(np.array(history.history["loss"]))
        return ppl

    def get_model(self):
        return self.model

    def evaluate_model(self,inputs):
        iter_data = []
        for i in range(len(inputs)):
            inputs[i].init_reader(K.get_session())
            iter_data.append(inputs[i].epoch_input())
            K.get_session().run(iter_data[i].initializer)
        x = tf.cast(iter_data[0].get_next()[0], tf.float32)
        y = tf.one_hot(iter_data[1].get_next()[3],self.config.vocab_size)
        loss = self.model.evaluate(x=x, y=y, steps=self.config.test_steps)[0]
        return np.exp(loss)

    def save_model(self,filename):
        self.model.save(filename)