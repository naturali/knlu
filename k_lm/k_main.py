import numpy as np
from k_lm.k_reader import RawStringDatasetReader
from k_lm.k_config import m_config
from k_lm.k_model import LanguageModel


def main():
    config = m_config
    data_reader_type = RawStringDatasetReader
    train_data_input = data_reader_type(m_config.data_dir,
                                  m_config.train_dir, is_train=True)
    train_data_target = data_reader_type(m_config.data_dir,
                                    m_config.train_dir, is_train=True)
    train_model = LanguageModel(config)

    valid_data_input = data_reader_type(m_config.data_dir,
                                  m_config.valid_dir, is_train=True)
    valid_data_target = data_reader_type(m_config.data_dir,
                                    m_config.valid_dir, is_train=True)

    test_data_input = data_reader_type(m_config.data_dir,
                                  m_config.test_dir, is_train=True)
    test_data_target = data_reader_type(m_config.data_dir,
                                    m_config.test_dir, is_train=True)


    k_model = train_model.build_lm_model([train_data_input, train_data_target])

    print(k_model.summary())
    history = k_model.fit(
        epochs=config.max_epoch,
        steps_per_epoch=config.steps_per_epoch)
    ppl = np.exp(np.array(history.history["loss"]))
    valid_ppl = train_model.evaluate_model([valid_data_input,valid_data_target],k_model)
    test_ppl = train_model.evaluate_model([test_data_input,test_data_target],k_model)
    for i in range(len(ppl)):
        print("epoch "+str(i+1)+" ppl:",ppl[i])
    print("valid's ppl:", valid_ppl)
    print("test's ppl:", test_ppl)

if __name__ == "__main__":
    main()
