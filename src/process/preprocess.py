# import sys
# print(sys.path)

from configuration.config import *


def preprocess():
    print("Preprocessing...")
    print('ROOT_DIR', ROOT_DIR)
    print('', RAW_DATA_DIR / RAW_TRAIN_DATA)


if __name__ == '__main__':
    preprocess()
