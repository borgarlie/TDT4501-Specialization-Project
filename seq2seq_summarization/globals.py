import torch
from torch.autograd import Variable
import time
import math
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker

PAD_token = 0
SOS_token = 1
EOS_token = 2

use_cuda = torch.cuda.is_available()

view_plot = False
single_char = False

# Preparing Training Data
# -----------------------
#
# To train, for each pair we will need an input tensor (indexes of the
# words in the input sentence) and target tensor (indexes of the words in
# the target sentence). While creating these vectors we will append the
# EOS token to both sequences.


# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    # print(seq)
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq


def indexes_from_sentence(vocabulary, sentence):
    if single_char:
        return [vocabulary.word2index[word] for word in sentence] + [EOS_token]
    return [vocabulary.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def variable_from_sentence(vocabulary, sentence):
    indexes = indexes_from_sentence(vocabulary, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variables_from_pair(article, title, vocabulary):
    input_variable = variable_from_sentence(vocabulary, article)
    target_variable = variable_from_sentence(vocabulary, title)
    return input_variable, target_variable


######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent, total_runtime):
    now = time.time()
    s = now - since + total_runtime
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs)), s

######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#


# def show_plot(points):
#     plt.figure()
#     fig, ax = plt.subplots()
#     # this locator puts ticks at regular intervals
#     loc = ticker.MultipleLocator(base=0.2)
#     ax.yaxis.set_major_locator(loc)
#     plt.plot(points)
#     if view_plot:
#         plt.show()
