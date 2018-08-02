from keras.models import Model
import numpy as np
import tensorflow as tf
import math
from keras.layers import Dense,Dropout,Input,Bidirectional
from keras.layers import Embedding,LSTM
from keras.optimizers import SGD
from keras import backend as K


class LanguageModel(object):
    def __init__(self,config):
        self.config = config

    def build_lm_model(self, train_data):
        iter_data = []
        for i in range(len(train_data)):
            train_data[i].init_reader(K.get_session())
            iter_data.append(train_data[i].epoch_input())
            K.get_session().run(iter_data[i].initializer)
        inputs = Input(tensor=tf.cast(iter_data[0].get_next()[0], tf.float32))
        embedding_layer = Embedding(self.config.vocab_size,
                                    self.config.embedding_size,
                                    input_length=self.config.num_steps - 1)
        X = embedding_layer(inputs)
        for _ in range(self.config.num_layers):
            X = Bidirectional(
                LSTM(
                    self.config.hidden_size,
                    activation='tanh',
                    recurrent_activation='hard_sigmoid',
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    recurrent_initializer='orthogonal',
                    bias_initializer='zeros',
                    return_sequences=True))(X)
            X = Dropout(0.1)(X)

        predictions = Dense(self.config.vocab_size, activation='softmax')(X)
        sgd = SGD(
            lr=self.config.learning_rate,
            momentum=self.config.sgd_momentum,
            decay=self.config.sgd_decay,
            nesterov=False)
        model = Model(inputs=inputs, outputs=predictions)
        # parallel_model = multi_gpu_model(model, gpus=2)
        # parallel_model = model # can be changed to multi_gpu_models
        model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'],
            target_tensors=[
                tf.one_hot(
                    iter_data[1].get_next()[3],
                    self.config.vocab_size)])
        return model




    def evaluate_model(self,train_data,k_model):
        iter_data = []
        for i in range(len(train_data)):
            train_data[i].init_reader(K.get_session())
            iter_data.append(train_data[i].epoch_input())
            K.get_session().run(iter_data[i].initializer)
        x = tf.cast(iter_data[0].get_next()[0], tf.float32)
        y = tf.one_hot(iter_data[1].get_next()[3],self.config.vocab_size)
        loss = k_model.evaluate(x=x, y=y, steps=self.config.test_steps)[0]
        return math.exp(loss)