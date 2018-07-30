import os
import tensorflow as tf
import reader
from config import m_config
from model import LanguageModel
from reader import RawStringDatasetReader


def main():
    train_file_dir = os.getcwd() + m_config.train_file
    data_reader_type = RawStringDatasetReader(train_file_dir)
    itera_x = data_reader_type.itera_x
    itera_y = data_reader_type.itera_y
    train_model = LanguageModel()
    print("ppl:", train_model.build_lm_model(itera_x, itera_y))


if __name__ == "__main__":
    main()
