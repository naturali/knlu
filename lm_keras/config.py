class ModelConfig(object):
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 40
        self.EMBEDDING_DIM = 100
        self.batch_size = 20
        self.units = 200
        self.nb_words = 9999
        self.nb_time_steps = 20
        self.nb_input_vector = 1
        self.sgd_lr = 1.0
        self.sgd_momentum = 0.9
        self.sgd_decay = 0.0
        self.nb_epoch = 13
        self.steps_per_epoch = 40000
        self.voc_num = 10000
        self.train_file = "/simple-examples/data/ptb.train.txt"
        self.valid_file = "/simple-examples/data/ptb.valid.txt"
        self.test_file = "/simple-examples/data/ptb.test.txt"
        self.valid_steps = 3370
        self.test_steps = 3761

m_config = ModelConfig()
