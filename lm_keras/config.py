class ModelConfig(object):
    def __init__(self):
        self.MAX_SEQUENCE_LENGTH = 20
        self.EMBEDDING_DIM = 100
        self.VALIDATION_SPLIT = 0.2
        self.batch_size = 20
        self.units = 200
        self.nb_words = 9999
        self.nb_time_steps = 20
        self.nb_input_vector = 1
        self.sgd_lr = 1.0
        self.sgd_momentum = 0.9
        self.sgd_decay = 0.0
        self.nb_epoch = 3
        self.steps_per_epoch = 25


m_config = ModelConfig()
