import unittest

from tensorflow import keras

from utils import lowercase_and_convert_to_ids, split_features_and_labels


class TestSum(unittest.TestCase):

    def test_lowercase_and_convert_to_ids(self):
        lookup_layer = keras.layers.StringLookup(
            vocabulary=["Aa", "Bb"]
        )
        result = lowercase_and_convert_to_ids(["Aa"], lookup_layer)
        self.assertEqual(result, 0)

    def test_split_features_and_labels(self):
        data = [(['1', '2'], [1, 0]), (['1', '2'], [1, 0])]
        train_data_text, tokens = split_features_and_labels(data)
        self.assertEqual(1 in tokens, True)
        self.assertEqual(0 in tokens, True)
        self.assertEqual(train_data_text[0], "1")
        self.assertEqual(train_data_text[1], "2")
        self.assertEqual(len(train_data_text), 4)


if __name__ == '__main__':
    unittest.main()
