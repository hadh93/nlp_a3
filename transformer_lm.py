# models.py
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch import optim

from transformer import PositionalEncoding


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        """
        Scores a bunch of characters following context. That is, returns
        log P(nc1, nc2, nc3, ... | context) = log P(nc1 | context) + log P(nc2 | context, nc1), ...
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param next_chars:
        :param context:
        :return: The float probability
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)


class NLModelEncoder(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, num_positions)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_internal)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.linear = nn.Linear(d_model, num_classes)

    def forward(self, indices):
        embedded = self.embedding(indices)
        embedded_with_positions = self.positional_encoding(embedded)

        mask = torch.triu(torch.ones(len(indices), len(indices)) * float('-inf'), diagonal=1)

        output = self.transformer_encoder(embedded_with_positions, mask=mask, is_causal=True)
        output = self.linear(output)
        return output


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, indexer):
        self.vocab_size = vocab_size
        self.encoder = NLModelEncoder(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
        self.indexer = indexer
        self.num_positions = num_positions

    def get_next_char_log_probs(self, context):
        if not context:
            return np.zeros(self.vocab_size)

        self.encoder.eval()
        with torch.no_grad():
            input_indices = [self.indexer.index_of(c) for c in context]
            input_indices = input_indices[-self.num_positions:]
            input_tensor = torch.tensor(input_indices, dtype=torch.long)
            input_tensor = input_tensor.unsqueeze(0)
#            print("Input tensor shape:", input_tensor.shape)
            output = self.encoder(input_tensor)
#            print("Output tensor shape:", output.shape)
            log_probs = nn.functional.log_softmax(output, dim=2)
#            print("Log probs tensor shape:", log_probs.shape)

        return np.asarray(log_probs[0, -1, :])

    def get_log_prob_sequence(self, next_chars, context):
        self.encoder.eval()
        log_prob = 0.0
        for char in next_chars:
            log_probs = self.get_next_char_log_probs(context)
            log_prob += log_probs[self.indexer.index_of(char)]
            context += char
        return log_prob


def divide_string(input_string, chunk_length=20):
    return [input_string[i:i + chunk_length] for i in range(0, len(input_string), chunk_length)]


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """

    vocab_size = len(vocab_index)
    num_positions = 20;
    d_model = 32;
    d_internal = 129
    num_classes = 27;
    num_layers = 2  # num_heads=?
    nlm = NeuralLanguageModel(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, vocab_index)
    model = nlm.encoder
    train_text_divided = divide_string(train_text)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ex_idxs = [i for i in range(len(train_text_divided))]
    random.shuffle(ex_idxs)
    num_epochs = 1
    for t in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        loss_fcn = nn.NLLLoss()
        idx = 0
        for ex_idx in ex_idxs:
            idx += 1
            if idx % 100 == 0:
                print(f"Epoch {t+1}: {idx} / {len(ex_idxs)}")
            chunk = train_text_divided[ex_idx]
            for k in range(1, num_positions + 1):
                padded_chunk = " " * (num_positions - k) + chunk[:k]
                output = model(torch.tensor([vocab_index.index_of(c) for c in padded_chunk], dtype=torch.long))
                target = torch.tensor([vocab_index.index_of(c) for c in chunk[:k]], dtype=torch.long)
                loss = loss_fcn(output, target)
                loss_this_epoch += loss.item()

                model.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch {t+1} Loss: {loss_this_epoch}")

    return nlm
