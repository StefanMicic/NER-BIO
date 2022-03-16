import os
import shutil
import unittest

import numpy as np

from data_preprocessing.word2vec import train_word2vec


class TestSum(unittest.TestCase):

    def setUp(self) -> None:
        os.mkdir("tmp_dir")

    def tearDown(self) -> None:
        shutil.rmtree('tmp_dir')

    def test_create_word2vec(self):
        cfg = {
            "data_preprocessing": {
                "input_data_path": "tmp_dir",
                "num_of_samples": 100
            }
        }
        x = np.array([['a', 'b'], ['a', 'c']])
        np.save("tmp_dir/train", x)
        os.rename("tmp_dir/train.npy", "tmp_dir/train.pkl")
        model = train_word2vec(cfg)
        self.assertEqual(len(model.wv['a']), 100)


if __name__ == '__main__':
    unittest.main()
