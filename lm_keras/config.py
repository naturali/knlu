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


m_config = ModelConfig()
