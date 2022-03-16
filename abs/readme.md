# Named Entity Recognition - Stefan Micic

## Approach

I presented two solutions for this problem. One using transformer model trained from scratch and the second one with
bi-directional RNN along with Word2Vec. (I though about using BERT for embeddings but it is said to me that using SOTA
things is not necessary).

## Explanation

Everything is configurable (config.json). Firstly, when running program `python app.py` you can add parameter to say if
you want to train transformer or rnn (`-t no`). In config.json you can decide whether to do the whole pipeline or just
one part. Easiest way to run everything is to used docker file or `run_application.sh`. There are two directories. One
for data preprocessing (for transformers) and for word2vec training, and the second dir for model architectures.

## Tests

I wrote a couple of tests, both unit and integration. To run unit test run `cd tests/unit && pytest .` of for
integration tests `cd integration/unit && pytest .`
