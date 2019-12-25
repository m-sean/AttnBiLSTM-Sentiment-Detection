#!/usr/bin/env python

"""Constructs a class which passes along data and attributes to the trainer and model."""

import torchtext
import logging


class DataLoader:
    def __init__(self, batch_size, label, preprocess, dataset, vectorizer):
        self.batch_size = batch_size
        self.label = label
        self.preprocess = preprocess
        self.train_data, self.test_data = dataset.splits(self.preprocess, self.label)
        self.preprocess.build_vocab(self.train_data, vectors=vectorizer)
        self.label.build_vocab(self.train_data)
        self.embeddings = self.preprocess.vocab.vectors

        logging.info(f"Length of Text Vocabulary: {self.getEmbeddingShape()[0]}")
        logging.info(f"Vector size of Text Vocabulary: {self.getEmbeddingShape()}")
        logging.info(f"Number of labels: {self.getNumLabels()}")

    def getEmbeddingShape(self):
        v, l = self.preprocess.vocab.vectors.size()
        return (v, l)

    def getNumLabels(self):
        return len(self.label.vocab)

    def split_iters(self):
        "Split off 30% of the training data for validation and return w/ test set."

        train_data, valid_data = self.train_data.split()
        train_iter, valid_iter, test_iter = torchtext.data.BucketIterator.splits(
            (train_data, valid_data, self.test_data),
            batch_size=self.batch_size,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True,
        )
        return (train_iter, valid_iter, test_iter)
