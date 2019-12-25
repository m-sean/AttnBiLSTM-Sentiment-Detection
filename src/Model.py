#!/usr/bin/env python

import torch

"""Constructs an LSTM model with attention and optional bidirectionality."""


class LSTMwAttention(torch.nn.Module):
    def __init__(
        self,
        batch_size: int,
        num_classes: int,
        hidden_size: int,
        vocab_size: int,
        dropout: float,
        bidirectional: bool,
        embedding_length: int,
        weights,
    ):
        super(LSTMwAttention, self).__init__()

        self.batch_size = batch_size
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional
        self.embedding_length = embedding_length
        self.weights = weights

        self.encoder = torch.nn.Embedding(vocab_size, embedding_length)
        self.encoder.weights = torch.nn.Parameter(weights, requires_grad=False)
        self.lstm = torch.nn.LSTM(
            embedding_length,
            hidden_size,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.w = torch.nn.Parameter(torch.randn(1, hidden_size, 1))
        self.fc = torch.nn.Linear(self.hidden_size, self.num_classes)

    def attention_mech(self, lstm_output, hid_n, inpt):
        "Computes attention output using matrix multiplication."
        batch_size, seq_len = inpt.shape
        hid_n = hid_n.view(
            1, self.bidirectional + 1, self.batch_size, self.hidden_size
        )[-1]
        hid_n = hid_n.permute(1, 0, 2)
        hid_n = hid_n.sum(dim=1)
        lstm_output = lstm_output.view(
            batch_size, seq_len, self.bidirectional + 1, self.hidden_size
        )
        lstm_output = lstm_output.sum(dim=2)
        att = torch.bmm(torch.tanh(lstm_output), self.w.repeat(batch_size, 1, 1))
        att = torch.nn.functional.softmax(att, dim=1)  # att(batch_size, seq_len, 1)
        att = torch.bmm(lstm_output.transpose(1, 2), att).squeeze(2)
        attn_output = torch.tanh(att)  # attn_output(batch_size, lstm_dir_dim)
        return attn_output

    def forward(self, inpt):
        batch_size, seq_len, *_ = inpt.shape
        inp = self.encoder(inpt)
        lstm_output, (hid_n, final_c) = self.lstm(inp)
        lstm_output = self.dropout(lstm_output)
        output = self.attention_mech(lstm_output, hid_n, inpt)
        logits = self.fc(output)
        return logits
