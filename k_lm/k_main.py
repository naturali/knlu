
import time
import keras
from k_lm.k_reader import RawStringDatasetReader
from k_lm.k_config import m_config
from k_lm.k_model import LanguageModel


def train_model():
    config = m_config
    data_reader_type = RawStringDatasetReader
    train_data_input = data_reader_type(m_config.data_dir,
                                  m_config.train_dir, is_train=True)
    train_data_target = data_reader_type(m_config.data_dir,
                                    m_config.train_dir, is_train=True)
    valid_data_input = data_reader_type(m_config.data_dir,
                                  m_config.valid_dir, is_train=False)
    valid_data_target = data_reader_type(m_config.data_dir,
                                    m_config.valid_dir, is_train=False)
    test_data_input = data_reader_type(m_config.data_dir,
                                  m_config.test_dir, is_train=False)
    test_data_target = data_reader_type(m_config.data_dir,
                                    m_config.test_dir, is_train=False)

    train_model = LanguageModel(config)
    train_model.build_lm_model()
    ppl = train_model.fit_lm_model([train_data_input, train_data_target])
    print(train_model.get_model().summary())
    valid_ppl = train_model.evaluate_model([valid_data_input,valid_data_target])
    test_ppl = train_model.evaluate_model([test_data_input,test_data_target])
    for i in range(len(ppl)):
        print("epoch "+str(i+1)+" ppl:",ppl[i])
    print("valid's ppl:", valid_ppl)
    print("test's ppl:", test_ppl)
    return train_model

def run_model():
    model = keras.models.load_model(m_config.load_filepath)
    data_reader_type = RawStringDatasetReader
    test_data_input = data_reader_type(m_config.data_dir,
                                  m_config.test_dir, is_train=True)
    test_data_target = data_reader_type(m_config.data_dir,
                                    m_config.test_dir, is_train=True)
    eval_model = LanguageModel(m_config)
    eval_ppl =  eval_model.evaluate_model([test_data_input,test_data_target],model)
    print("eval's pll:", eval_ppl)
    return eval_model

def save_model():
    # model = train_model()
    # config = m_config
    # save_dir = config.save_dir
    # str_time = time.asctime(time.localtime(time.time()))
    # save_name = "keras_model "+str_time+".h5"
    # file_path = save_dir+save_name
    # print("model has saved in " + file_path)
    # model.save_model(file_path)
    pass # can't use because  Subclassed networks are not serializable in keras 2.2.0


def main():
    if m_config.mode == 'train':
        train_model()
    if m_config.mode == 'run':
        run_model()
    if m_config.mode == 'save':
        save_model()



if __name__ == "__main__":
    main()
