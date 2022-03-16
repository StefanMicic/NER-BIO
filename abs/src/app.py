import argparse
import json
import os
from itertools import compress
from typing import Dict

import numpy as np
from loguru import logger as log
from sklearn.metrics import classification_report
from tensorflow import keras

from data_preprocessing.preprocessing import DataPreprocessing
from models.ner_model import NERModel
from models.rnn import RNNModel


def prepare_data(input_data_path: str, prepared_data_path: str, vocab_size: int, num_of_samples: int):
    log.info("Preprocessing step")
    train_data = np.load(f'{input_data_path}/train.pkl', allow_pickle=True)
    test_data = np.load(f'{input_data_path}/test.pkl', allow_pickle=True)

    data_preprocessing_object = DataPreprocessing(train_data, vocab_size)
    if not os.path.exists(prepared_data_path):
        os.mkdir(prepared_data_path)
        data_preprocessing_object(f"{prepared_data_path}/train", train_data, num_of_samples)
        data_preprocessing_object(f"{prepared_data_path}/test", test_data, num_of_samples)


def train_model(x: np.array, y: np.array, training_config: Dict) -> NERModel:
    log.info("Training step")
    ner_model = NERModel(training_config['num_of_labels'] + 1, training_config['vocab_size'] + 1,
                         maxlen=max([len(s) for s in x]), embed_dim=32, num_heads=4,
                         ff_dim=64)
    ner_model.compile(optimizer="adam", metrics=['accuracy'],
                      loss=keras.losses.SparseCategoricalCrossentropy())
    ner_model.fit(x, y, epochs=training_config['epochs'], batch_size=training_config['batch_size'],
                  validation_split=0.2)
    return ner_model


def evaluate_model(ner_model: NERModel, x_test: np.array, y_test: np.array):
    log.info("Evaluation step")
    p = ner_model.predict(x_test)
    y_pred = []
    y_true = []
    for t, prediction in zip(y_test, p):
        mask = [n != 63 for n in t]
        y_true.extend(list(compress(t, mask)))
        y_pred.extend(list(compress(np.argmax(prediction, axis=-1), mask)))
    print(classification_report(y_true, y_pred, zero_division=0))


def pipeline(arguments):
    if arguments.transformer == 'yes':
        log.info("Transformer model")
        if cfg['data_preprocessing']['apply'] == 'yes':
            prepare_data(cfg['data_preprocessing']['input_data_path'],
                         cfg['data_preprocessing']['prepared_data_path'],
                         cfg['training']['vocab_size'],
                         cfg['data_preprocessing']['num_of_samples'])
        if cfg['training']['apply'] == 'yes':
            prepared_data_path = cfg['data_preprocessing']['prepared_data_path']
            x_train = np.load(f'{prepared_data_path}/train_x.npy')
            y_train = np.load(f'{prepared_data_path}/train_y.npy')
            model = train_model(x_train, y_train, cfg['training'])

            evaluate_model(model, np.load(f'{prepared_data_path}/test_x.npy'),
                           np.load(f'{prepared_data_path}/test_y.npy'))
            model.save('transformer_model')
    else:
        log.info("RNN model")
        rnn_model = RNNModel(cfg)
        rnn_model.preprocessing()
        embedding_matrix = rnn_model.create_embed_matrix()
        model = rnn_model.create_model(embedding_matrix)
        rnn_model.train_rnn(model).save('bi_rnn_model')


if __name__ == "__main__":
    with open('config.json') as json_file:
        cfg = json.load(json_file)

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="config.json")
    parser.add_argument("-t", "--transformer", type=str, default='yes')
    args = parser.parse_args()
    pipeline(args)
