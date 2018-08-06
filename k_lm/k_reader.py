import random
import os
import tensorflow as tf
from k_lm.k_config import m_config
from util.file_util import walkdir

SOS = '<sos>'
EOS = '<eos>'
UNK = '<unk>'


class TFDatasetReader(object):
    def __init__(self, dir, folders, is_train, extension):
        self.batch_size = m_config.train_batch_size if \
            is_train else m_config.valid_batch_size
        self.is_train = is_train
        self.dir = dir
        self.folders = folders
        self.filenames = []
        self.gpu_num = m_config.gpu_num
        print("folders", folders)
        print("reader batch size:", self.batch_size)
        print("reader num_steps:", m_config.num_steps)
        for f in folders.split('|'):
            if len(f) > 0:
                self.filenames += walkdir(os.path.join(dir, f), extension=extension)
        self.num_steps = m_config.num_steps
        self.shuffle_buffer_size = self.batch_size * m_config.shuffle_batch_num
        self.prefetch_buffer_size = self.batch_size * m_config.prefetch_batch_num
        self.word_dict = {}
        with open(m_config.dict_path, 'r') as f:
            for l in f.readlines():
                word, idx = l.split(' ')[0], l.split(' ')[1]
                self.word_dict[word] = int(idx)
        print("word_dict size:", len(self.word_dict))
        self.SOS_ID = self.word_dict[SOS]
        self.EOS_ID = self.word_dict[EOS]
        self.UNK_ID = self.word_dict[UNK]

        if m_config.use_word_freq:
            self.word_freq = {}
            with open(m_config.word_freq_path, 'r') as f:
                for l in f.readlines():
                    word, freq = l.strip().split("\t")
                    self.word_freq[self.word_dict[word]] = float(freq)
            self.word_freq_list = \
                [self.word_freq[self.word_dict[UNK]]] * len(self.word_dict)
            for idx, freq in self.word_freq.items():
                self.word_freq_list[idx] = freq
            self.word_freq_list = tf.constant(
                self.word_freq_list, dtype=tf.float32)

    def preprocess_function(self, x):
        inputs = tf.concat([[self.SOS_ID], x, [self.EOS_ID]], axis=0)
        bw_inputs = tf.reverse_sequence(
            tf.expand_dims(inputs, 0),
            tf.shape(inputs),
            seq_dim=1,
            batch_axis=0)
        bw_inputs = tf.squeeze(bw_inputs, 0)
        input_length = tf.shape(inputs)[0] - 1
        targets = tf.slice(inputs, [1], [input_length])
        bw_targets = tf.slice(bw_inputs, [1], [input_length])
        inputs = tf.slice(inputs, [0], [input_length])
        bw_inputs = tf.slice(bw_inputs, [0], [input_length])
        if m_config.use_word_freq:
            targets_freq = tf.gather(self.word_freq_list, targets)
            bw_targets_freq = tf.gather(self.word_freq_list, bw_targets)
            return inputs, bw_inputs, input_length, targets, bw_targets, \
                   targets_freq, bw_targets_freq
        else:
            return inputs, bw_inputs, input_length, targets, bw_targets

    def padded_batch_function(self, dataset):
        if m_config.use_word_freq:
            return dataset.padded_batch(
                self.batch_size,
                ([None], [None], [], [None], [None], [None], [None]),
                (tf.ones([], tf.int64) * self.EOS_ID,
                 tf.ones([], tf.int64) * self.SOS_ID,
                 tf.zeros([], tf.int32),
                 tf.ones([], tf.int64) * self.EOS_ID,
                 tf.ones([], tf.int64) * self.SOS_ID,
                 tf.zeros([], tf.float32),
                 tf.zeros([], tf.float32))).filter(
                lambda a, b, c, d, e, f, g: tf.equal(tf.shape(a)[0], self.batch_size))
        else:
            return dataset.padded_batch(
                self.batch_size,
                ([None], [None], [], [None], [None]),
                (tf.ones([], tf.int64) * self.EOS_ID,
                 tf.ones([], tf.int64) * self.SOS_ID,
                 tf.zeros([], tf.int32),
                 tf.ones([], tf.int64) * self.EOS_ID,
                 tf.ones([], tf.int64) * self.SOS_ID)).filter(
                lambda a, b, c, d, e: tf.equal(tf.shape(a)[0], self.batch_size))

    def epoch_input(self):
        return NotImplemented

    def init_reader(self, sess):
        pass

class RawStringDatasetReader(TFDatasetReader):

    def __init__(self, dir, folders, is_train):

        if m_config.raw_data_tokenizer == "utf-8":
            import plugins.tokenize_utf8.tokenize_utf8_ops as tokenize_utf8_ops
            tokenize_ops = tokenize_utf8_ops
        elif m_config.raw_data_tokenizer == "space":
            import plugins.tokenize_space.tokenize_space_ops as tokenize_space_ops
            tokenize_ops = tokenize_space_ops
        else:
            import plugins.tokenize_ch_en.tokenize_ch_en_ops as tokenize_ch_en_ops
            tokenize_ops = tokenize_ch_en_ops

        super().__init__(dir, folders, is_train, m_config.raw_data_extension)
        kv_initializer = tf.contrib.lookup.TextFileInitializer(
            m_config.dict_path, tf.string, 0, tf.int64, 1, delimiter=" ")
        self.lookup_table = tf.contrib.lookup.HashTable(kv_initializer, self.UNK_ID)

        filenames = self.filenames
        if self.is_train:
            random.shuffle(filenames)
        dataset = tf.data.TextLineDataset(filenames)
        if self.is_train:
            dataset.shuffle(self.shuffle_buffer_size)

        dataset = dataset.map(lambda text: tokenize_ops.tokenize([text]), num_parallel_calls=self.gpu_num * 4).repeat()
        dataset = dataset.map(lambda tkns: self.lookup_table.lookup(tkns), num_parallel_calls=self.gpu_num * 4)
        dataset = dataset.filter(lambda x: tf.shape(x)[0] + 2 <= self.num_steps)
        dataset = dataset.map(self.preprocess_function, num_parallel_calls=self.gpu_num * 4)
        dataset = self.padded_batch_function(dataset)
        dataset = dataset.prefetch(self.prefetch_buffer_size)
        self.iterator = dataset.make_initializable_iterator()

    def init_reader(self, sess):
        sess.run(self.lookup_table.init)

    def epoch_input(self):
        return self.iterator
