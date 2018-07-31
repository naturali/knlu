import os
import reader
import math
from config import m_config
from model import LanguageModel
from reader import RawStringDatasetReader


def main():
    train_file_dir = os.getcwd() + m_config.train_file
    valid_file_dir = os.getcwd() + m_config.valid_file
    test_file_dir = os.getcwd()  + m_config.test_file

    data_train_reader_type = RawStringDatasetReader(train_file_dir)
    itera_x_train = data_train_reader_type.itera_x
    itera_y_train = data_train_reader_type.itera_y

    data_valid_reader_type = RawStringDatasetReader(valid_file_dir)
    itera_x_valid = data_valid_reader_type.itera_x
    itera_y_valid = data_valid_reader_type.itera_y

    data_test_reader_type = RawStringDatasetReader(test_file_dir)
    itera_x_test = data_test_reader_type.itera_x
    itera_y_test = data_test_reader_type.itera_y

    train_model = LanguageModel()
    k_model,ppl = train_model.build_lm_model(itera_x_train, itera_y_train)
    valid_loss = k_model.evaluate(x=itera_x_valid, y=itera_y_valid,steps=m_config.valid_steps)[0]
    test_loss = k_model.evaluate(x=itera_x_test, y=itera_y_test,steps=m_config.test_steps)[0]
    print("trainning's ppl:", ppl)
    print("valid's ppl:", math.exp(valid_loss))
    print("test's ppl:", math.exp(test_loss))


if __name__ == "__main__":
    main()
