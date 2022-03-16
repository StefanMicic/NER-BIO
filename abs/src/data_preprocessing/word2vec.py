from typing import Dict

import nltk
import numpy as np
from gensim.models import Word2Vec
from loguru import logger as log

nltk.download("stopwords")


def train_word2vec(cfg: Dict) -> Word2Vec:
    log.info("Word2Vec is being trained!")
    input_data_path = cfg['data_preprocessing']['input_data_path']
    train_data = np.load(f'{input_data_path}/train.pkl', allow_pickle=True)
    text = []
    for record in train_data:
        text.append([x.lower() for x in record[0]])

    return Word2Vec(sentences=text[:cfg['data_preprocessing']['num_of_samples']], window=4, min_count=1, workers=4)
