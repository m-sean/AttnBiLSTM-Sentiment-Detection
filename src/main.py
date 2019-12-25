#!/usr/bin/env python

import argparse
import collections
import functools
import logging
import torch
import torchtext

from DataLoader import DataLoader
from Model import LSTMwAttention
from Trainer import ModelTrainer


def main(args):

    # load the data and split
    data_loader = DataLoader(
        batch_size=args.batch_size,
        preprocess=torchtext.data.Field(
            sequential=True,
            tokenize=functools.partial(str.split),
            lower=True,
            include_lengths=True,
            batch_first=True,
            fix_length=200,
        ),
        label=torchtext.data.LabelField(),
        dataset=torchtext.datasets.IMDB,
        vectorizer=torchtext.vocab.GloVe(name="6B", dim=args.embed_size),
    )
    train_iter, valid_iter, test_iter = data_loader.split_iters()

    # instantiate the model
    model = LSTMwAttention(
        batch_size=data_loader.batch_size,
        bidirectional=args.bidirectional,
        num_classes=data_loader.getNumLabels(),
        hidden_size=args.hidden_size,
        vocab_size=data_loader.getEmbeddingShape()[0],
        dropout=args.dropout,
        embedding_length=data_loader.getEmbeddingShape()[1],
        weights=data_loader.embeddings,
    )

    # instantiate a trainer, avoid showing Adam a momentum hyperparameter
    if args.optimizer.lower() != "adam":
        trainer = ModelTrainer(
            model,
            modelname=args.modelname,
            grad_clipping=args.clip,
            earlystop=args.earlystop,
            optimizer=args.optimizer.lower(),
            lr=args.lr,
        )
    else:
        trainer = ModelTrainer(
            model,
            modelname=args.modelname,
            grad_clipping=args.clip,
            earlystop=args.earlystop,
            optimizer=args.optimizer.lower(),
            lr=args.lr,
            momentum=args.momentum if args.momentum else 0.9,
        )

    # skip training if we just want to evaluate on the test set
    if args.test:
        test_loss, test_acc = trainer.validate(test_iter, istrain=False)
        logging.info(f"<EVALUATION>\tLoss: {test_loss:.5f}\tAcc: {test_acc:.5f}")

    else:
        for epoch in trainer.train(train_iter, epochs=args.epochs):
            train_loss, train_acc = epoch
            val_loss, val_acc = trainer.validate(valid_iter)
            logging.info(
                f"Train Loss: {train_loss:.5f}\tTrain Acc: {train_acc:.5f}\tVal. Loss: {val_loss:5f}\tVal. Acc: {val_acc:.5f}"
            )

        if trainer.earlystop:
            logging.info("Loading best model.")
            trainer.getbest()

        test_loss, test_acc = trainer.validate(test_iter, istrain=False)
        logging.info(f"<EVALUATION>\tLoss: {test_loss:.5f}\tAcc: {test_acc:.5f}")


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="number of samples per batch, default=32",
    )
    parser.add_argument(
        "--bidirectional",
        action="store_true",
        help="flag to make a BiLSTM, recommended",
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=0.1,
        help="amount for gradient clipping, default=0.1",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="fights overfitting with lstm dropout, default=30",
    )
    parser.add_argument(
        "--earlystop",
        action="store_true",
        help="use best model from training epochs, not always last, recommended",
    )
    parser.add_argument(
        "--embed_size",
        type=int,
        default=300,
        help="sets length of embedding vectors, default=300",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="number of epochs for training, default=20",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="number of neurons in the hidden LSTM layer, default=512",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="learning rate for training, default=0.0001",
    )
    parser.add_argument(
        "--modelname",
        type=str,
        default="model.pt",
        help="name of model for saving, default='model.pt'",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        help="momentum hyperparameter, applies only for RMSprop or SGD optimizers, default=0.9",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="rmsprop",
        help="optimizer to use for training, default='rmsprop'",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="evaluate an existing model, --modelname must be provided",
    )
    main(parser.parse_args())
