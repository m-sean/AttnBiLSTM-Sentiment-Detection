#!/usr/bin/env python

"""Contructs an object used to train and validate the model."""

import collections
import logging
import os
import torch


class ModelTrainer:
    def __init__(
        self,
        model,
        modelname,
        grad_clipping=0.1,
        earlystop=True,
        optimizer="rmsprop",
        **opt_kwargs,
    ):
        self.model = model
        self.modelname = modelname
        self.grad_clipping = grad_clipping
        self.earlystop = earlystop

        self.history = collections.defaultdict(list)
        self.history["acc"] = list()
        self.history["loss"] = list()
        if optimizer == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), **opt_kwargs)
        elif optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), **opt_kwargs)
        else:
            self.optimizer = torch.optim.RMSprop(model.parameters(), **opt_kwargs)

    @staticmethod
    def norm_clip(model, clip_value):
        "Fight vanising gradient by clipping with a given value."
        params = list(filter(lambda p: p.grad is not None, model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def getbest(self):
        "Load the best (if early stopping) model from the directory."
        self.model = torch.load(f"../model/{self.modelname}")

    def train(self, train_iter, epochs):
        "An iterator that yeilds loss and accuracy for each epoch of training."
        # make sure we have a safe place to store a trained model
        if not os.path.isdir("../model/"):
            os.mkdir("../model/")
        for e in range(epochs):
            # use cuda if we have it
            if torch.cuda.is_available():
                self.model.cuda()
                self.model.train()
            # keep track of batch loss and accuracy for averaging later
            total_epoch_loss = 0
            total_epoch_acc = 0
            steps = 0
            logging.info(f"<EPOCH {e+1}>")
            for idx, batch in enumerate(train_iter):
                texts = batch.text[0]
                targets = torch.autograd.Variable(batch.label).long()
                if torch.cuda.is_available():
                    texts = texts.cuda()
                    targets = targets.cuda()
                # check batch dimension, ignore any that don't fit
                if texts.size()[0] is not train_iter.batch_size:
                    continue
                self.optimizer.zero_grad()
                predictions = self.model(texts)
                loss = torch.nn.functional.cross_entropy(predictions, targets)
                num_correct = (
                    (
                        torch.max(predictions, 1)[1].view(targets.size()).data
                        == targets.data
                    )
                    .float()
                    .sum()
                )
                acc = num_correct / len(batch)
                loss.backward()
                self.norm_clip(self.model, self.grad_clipping)
                self.optimizer.step()
                steps += 1
                # show progress every 100 batches
                if steps % 100 == 0:
                    logging.info(
                        f"Batch {steps}\tLoss: {loss.item():.5f}\tAcc: {acc.item(): .5f}"
                    )

                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()

            yield total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)

    def validate(self, val_iter, istrain=True):
        "Evaluate trained model on development or test sets."
        total_loss = 0
        total_acc = 0
        self.model.eval()
        with torch.no_grad():
            for idx, batch in enumerate(val_iter):
                texts = batch.text[0]
                if texts.size()[0] is not val_iter.batch_size:
                    continue
                targets = torch.autograd.Variable(batch.label).long()
                if torch.cuda.is_available():
                    texts = texts.cuda()
                    targets = targets.cuda()
                predictions = self.model(texts)
                loss = torch.nn.functional.cross_entropy(predictions, targets)
                num_correct = (
                    (
                        torch.max(predictions, 1)[1].view(targets.size()).data
                        == targets.data
                    )
                    .float()
                    .sum()
                )
                acc = num_correct / len(batch)
                total_loss += loss.item()
                total_acc += acc.item()
        self.history["acc"].append(total_acc / len(val_iter))

        # if early stopping, only save the model if accuracy improves
        if (
            self.earlystop
            and total_acc / len(val_iter) >= max(self.history["acc"])
            and istrain
        ):
            logging.info(
                f"New best validation acc, saving model to '../model/{self.modelname}'"
            )
            torch.save(self.model, f"../model/{self.modelname}")
        elif istrain:
            torch.save(self.model, f"../model/{self.modelname}")
        return total_loss / len(val_iter), total_acc / len(val_iter)
