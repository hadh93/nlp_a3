import math
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
        self.log_softmax = nn.LogSoftmax()

    def forward(self, indices):
        embedded = self.embedding(indices)
        embedded_with_positions = self.positional_encoding(embedded)

        mask = torch.triu(torch.ones(len(indices), len(indices)) * float('-inf'), diagonal=1)

        output = self.transformer_encoder(embedded_with_positions, mask=mask, is_causal=True)
        output = self.linear(output)
        return self.log_softmax(output)


class NeuralLanguageModel(LanguageModel):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers, indexer):
        self.vocab_size = vocab_size
        self.encoder = NLModelEncoder(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)
        self.indexer = indexer
        self.num_positions = num_positions

    def get_next_char_log_probs(self, context):
        if not context:
            return np.zeros(self.vocab_size)
        context = " " * (self.num_positions - len(context)) + context
        with torch.no_grad():
            input_indices = [self.indexer.index_of(c) for c in context]
            input_tensor = torch.tensor(input_indices, dtype=torch.long)
            output = self.encoder(input_tensor)
            output = output[-1,:]

        return np.asarray(output)

    def get_log_prob_sequence(self, next_chars, context):
        log_prob = 0.0
        context = " "* (self.num_positions - len(context)) + context
        for char in next_chars:
            # print(f"Chunk context: ({context}), Gold: {char}, prob:{log_prob}")
            log_probs = self.get_next_char_log_probs(context)
            log_prob += log_probs[self.indexer.index_of(char)]
            context = context[1:]+char
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
    input_arr_all = []
    output_arr_all = []
    for ex_idx in range(len(train_text_divided)):
        input_arr = []
        output_arr = []
        if ex_idx != len(train_text_divided) - 1:
            chunk = train_text_divided[ex_idx] + train_text_divided[ex_idx + 1][:5]

            for k in range(0, num_positions + 5):
                tmp = " " * (num_positions - k) + chunk[:k]
                if len(tmp) > num_positions:
                    tmp = tmp[-num_positions:]
                input_arr.append(tmp)
            output_arr = input_arr[1:]
            output_arr.append(chunk[-num_positions:])
        else:
            chunk = train_text_divided[ex_idx]
            for k in range(num_positions):
                tmp = " " * (num_positions - k) + chunk[:k]
                input_arr.append(tmp)
            output_arr = input_arr[1:]
            output_arr.append(chunk[-num_positions:])
        input_tensor = []
        output_tensor = []
        for sentence in input_arr:
            input_tensor.append(torch.tensor([vocab_index.index_of(c) for c in sentence], dtype=torch.long))
        for sentence in output_arr:
            output_tensor.append(torch.tensor([vocab_index.index_of(c) for c in sentence], dtype=torch.long))

        input_arr_all.append(input_tensor)
        output_arr_all.append(output_tensor)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    ex_idxs = [i for i in range(len(train_text_divided))]
    num_epochs = 3
    for t in range(num_epochs):
        random.seed(t)
        loss_this_epoch = 0.0
        random.shuffle(ex_idxs)

        loss_fcn = nn.NLLLoss()
        idx = 0
        for ex_idx in ex_idxs:
            idx += 1
            """
            if idx % 1000 == 0:
                print(f"Epoch {t+1}: {idx} / {len(ex_idxs)}")
                print(f"Epoch {t + 1} Loss: {loss_this_epoch}")
                log_prob = nlm.get_log_prob_sequence(dev_text, "")
                avg_log_prob = log_prob / len(dev_text)
                perplexity = np.exp(-log_prob / len(dev_text))
                print(f"log_prob: {log_prob}, avg_log_prob: {avg_log_prob}, perplexity: {perplexity}")
            """
            input_tensor = input_arr_all[ex_idx]
            output_tensor = output_arr_all[ex_idx]

            for i in range(len(input_tensor)):

                output = model(input_tensor[i])
                target = output_tensor[i]

                loss = loss_fcn(output, target)
            loss_this_epoch += loss.item()
            model.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return nlm
