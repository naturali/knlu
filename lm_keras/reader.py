
import sys
import collections
import tensorflow as tf
from config import m_config

Py3 = sys.version_info[0] == 3


class TFDatasetReader(object):
    def __init__(self, dir):
        self.dir = dir

    def _read_words(self, filename):
        with tf.gfile.GFile(filename, "r") as f:
            if Py3:
                return f.read().replace("\n", "<eos>").split()
            else:
                return f.read().decode("utf-8").replace("\n", "<eos>").split()

    def _build_vocab(self, filename):
        data = self._read_words(filename)
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        word_to_id = dict(zip(words, range(len(words))))
        return word_to_id

    def _file_to_word_ids(self, filename, word_to_id):
        data = self._read_words(filename)
        return [word_to_id[word] for word in data if word in word_to_id]

    def preprocess_function(self, x):
        inputs = tf.concat([x], axis=0)
        input_length = tf.shape(inputs)[0] - 1
        inputs = tf.slice(inputs, [0], [input_length])
        return inputs

    def preprocess_target_function(self, x):
        inputs = tf.concat([x], axis=0)
        input_length = tf.shape(inputs)[0] - 1
        targets = tf.slice(inputs, [1], [input_length])
        return targets


class RawStringDatasetReader(TFDatasetReader):
    def __init__(self, dir):
        super().__init__(dir)
        data_set = tf.data.TextLineDataset(self.dir)
        word_to_id = self._build_vocab(self.dir)
        word_to_id_list = tf.constant([word for word in word_to_id])
        table = tf.contrib.lookup.index_table_from_tensor(
            mapping=word_to_id_list, num_oov_buckets=1, default_value=-1)
        data_set = data_set.map(
            lambda x: tf.string_split([x], delimiter=' ').values).repeat()
        data_set = data_set.map(
            lambda x: table.lookup(x))
        data_set = data_set.filter(
            lambda x: tf.shape(x)[0] +
            2 <= m_config.MAX_SEQUENCE_LENGTH)
        data_set = data_set.map(
            lambda x: tf.cast(x, tf.float32))
        data_set_x = data_set.map(self.preprocess_function)
        data_set_y = data_set.map(self.preprocess_target_function)
        data_set_x = data_set_x.padded_batch(1, m_config.MAX_SEQUENCE_LENGTH)
        data_set_y = data_set_y.padded_batch(1, m_config.MAX_SEQUENCE_LENGTH)
        data_set_x = data_set_x.map(lambda x: tf.reshape(
            x, [-1, m_config.nb_time_steps]))
        data_set_y = data_set_y.map(lambda x: tf.reshape(
            x, [-1, m_config.nb_time_steps]))
        data_set_y = data_set_y.map(
            lambda x: tf.cast(
                x,
                tf.int64))
        data_set_y = data_set_y.map(lambda x: tf.one_hot(x, m_config.voc_num))
        self.itera_x = data_set_x.make_initializable_iterator()
        self.itera_y = data_set_y.make_initializable_iterator()
