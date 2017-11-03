import random

import torch.nn as nn
import torch.nn.functional as F
import torch

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.

######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=1, batch_size=1, num_categories=5):
        super(DecoderRNN, self).__init__()
        self.batch_size = batch_size
        self.n_layers = n_layers

        # split to embed size and actual hidden size
        self.embed_size = hidden_size - num_categories
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, self.embed_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax()

    def forward(self, input, hidden, categories, batch_size=1):
        output = self.embedding(input).view(1, batch_size, self.embed_size)
        # print("OUTPUT = ", flush=True)
        # print(output, flush=True)

        categories = categories.unsqueeze(0)
        # print("categories: ", flush=True)
        # print(categories, flush=True)

        # concatenate embedding with categories
        output = torch.cat((output, categories), 2)

        # if random.random() < 0.01:
        #     print("OUTPUT 2 = ", flush=True)
        #     print(output, flush=True)
        #     # exit()

        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        result = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, n_layers=1, dropout_p=0.1, batch_size=1, num_categories=5):
        super(AttnDecoderRNN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.batch_size = batch_size

        # split to embed size and actual hidden size
        self.embed_size = hidden_size - num_categories
        self.hidden_size = hidden_size

        # Any other changes apart from embed size ?
        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, self.n_layers)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs, categories, batch_size=1):
        embedded = self.embedding(input).view(1, batch_size, self.embed_size)
        embedded = self.dropout(embedded)

        categories = categories.unsqueeze(0)
        embedded = torch.cat((embedded, categories), 2)

        cat = torch.cat((embedded[0], hidden[0]), 1)
        temp = self.attn(cat)
        attn_weights = F.softmax(temp)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.transpose(0, 1))

        output = torch.cat((embedded[0], attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def init_hidden(self):
        result = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
