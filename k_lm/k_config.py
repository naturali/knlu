import os
import sys
import tensorflow as tf

from util.config import ConfigParser


class ModelConfig(ConfigParser):
    def __init__(self, args=[]):
        super().__init__()
        self.async_training = False
        # path params
        self.data_dir = '/mnt/cephfs/dataset/ptb'
        self.restore_path = ''
        self.save_dir = '/home/sunzewen/save/'
        self.load_model = ""
        self.use_float16 = False
        self.restore_model = False
        self.trans_embedding = True
        self.use_treemax = False
        self.bi_direct_lm = True
        self.fused_rnn = True
        self.treemax_stddev = 1.0
        self.multiply_S = False
        self.variational_dropout = False
        self.pivots_size = 1023
        self.sampler_softmax = True
        self.relu_embedding = False
        self.init_zero_embedding = False
        self.quantize_graph = False
        self.quant_embedding = False
        self.quant_embedding_min = -8.0
        self.quant_embedding_max = 8.0
        self.epsilon_embedding_is_0 = 1e-3
        self.mask_small_embedding = False
        self.embedding_output_normalize = False
        self.embedding_att_temperature = False
        self.embedding_att_multi_head_num = 1
        self.gumble_gate_temperature = 0.1
        self.init_scale = 0.1
        self.max_grad_norm = 1.0
        self.num_layers = 3
        self.keep_prob = 0.8
        self.num_steps = 20
        self.vocab_size = 793471
        self.learning_rate = 0.2
        self.max_epoch = 2
        self.embedding_size = 512
        self.train_batch_size = 256
        self.debug_timeline = False
        self.valid_batch_size = 16
        self.hidden_size = 512
        self.cell_type = 'lstm'
        self.gpu_num = 1
        self.optimizer_name = 'adagrad'
        self.ww_avg_grad = False
        self.regularize = ''
        self.embedding_type = ''
        self.use_sampler_regularity = True
        self.regu_rate = 0.5
        self.num_sampled = 8192
        self.label_smoothing = 0.0
        self.mode = 'train'
        self.show_step = 100
        self.valid_epoch = 1
        self.test_embedding_step = 1000
        self.test_embedding_as_tree = False
        self.token_freq_path = ''
        self.wordsim_path = "/mnt/cephfs/dataset/cleaned-1-billion" \
                            "/mnt/cleaned/test_task/eval-word-vectors/"
        self.test_wordsim = False
        self.dict_file = '/mnt/cephfs/dataset/sentence_with_punc/ch_en_15k.dict'
        self.word_freq_file = 'local_mess/word_count_afterfunc'
        self.use_word_freq = False
        self.train_dir = 'train'
        self.valid_dir = 'valid'
        self.test_dir = 'test'
        self.dataset_type = 'raw'
        self.raw_data_tokenizer = 'ch_en'
        self.raw_data_extension = 'txt'
        self.test_words = "狗|男|花|跑|美"
        self.shuffle_batch_num = 8
        self.prefetch_batch_num = 32
        self.steps_per_epoch = 1000
        self.sgd_momentum = 0.1
        self.sgd_decay = 0.1
        self.test_steps = 1000
        self._parse_args(args)
        self.dict_path = os.path.join(self.data_dir, self.dict_file)
        self.word_freq_path = os.path.join(self.data_dir, self.word_freq_file)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        self.embedding_path = os.path.join(self.save_dir, 'embedding.pkl')
        if os.path.exists(self.dict_path):
            with open(self.dict_path) as f:
                self.vocab_size = len(f.readlines())
        self.data_type = tf.float16 if self.use_float16 else tf.float32


m_config = ModelConfig(sys.argv)
