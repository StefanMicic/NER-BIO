import json
import os.path
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from models.rnn import RNNModel


class TestSum(unittest.TestCase):
    def setUp(self) -> None:
        os.mkdir("tmp_dir")
        data = [(['1', '2'], [1, 0]), (['1', '2'], [1, 0])]
        np.save("tmp_dir/train", np.array(data))
        os.rename("tmp_dir/train.npy", "tmp_dir/train.pkl")

    def tearDown(self) -> None:
        shutil.rmtree('bi_rnn_model')
        shutil.rmtree('tmp_dir')

    @patch('models.rnn.RNNModel.train_rnn')
    def test_rnn_training(self, mock_api_call):
        with open('fixtures/config.json') as json_file:
            cfg = json.load(json_file)

        rnn_model = RNNModel(cfg)
        rnn_model.preprocessing()
        embedding_matrix = rnn_model.create_embed_matrix()
        model = rnn_model.create_model(embedding_matrix)

        mock_api_call.return_value = model
        rnn_model.train_rnn(model).save('bi_rnn_model')
        self.assertEqual(os.path.exists("bi_rnn_model"), True)


if __name__ == '__main__':
    unittest.main()
