import random

import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from classifier.cnn_classifier import CNN_Text
from seq2seq_summarization.globals import *
from seq2seq_summarization import preprocess as preprocess


# Train one batch
def train(categories, sequences, batch_size, model, optimizer, criterion):
    optimizer.zero_grad()
    categories_scores = model(sequences)

    category_batch_list = []
    for batch in range(0, batch_size):
        batch_cat = []
        for cat in categories[batch]:
            batch_cat.append(int(cat))
        category_batch_list.append(batch_cat)

    categories = Variable(torch.FloatTensor(category_batch_list))
    if use_cuda:
        categories = categories.cuda()

    loss = criterion(categories_scores, categories)

    loss.backward()
    optimizer.step()
    return loss.data[0]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def train_iters(articles, titles, vocabulary, model, optimizer):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    n_epochs = 10
    batch_size = 1
    print_every = 100
    plot_every = 100

    criterion = nn.BCEWithLogitsLoss()  # TODO: Could change to BCEwithLogitsLoss

    num_batches = int(len(articles) / batch_size)
    n_iters = num_batches * n_epochs

    lowest_loss = 999
    total_runtime = 0.0

    print("Starting training", flush=True)
    for epoch in range(1, n_epochs + 1):
        print("Starting epoch: %d" % epoch, flush=True)
        batch_loss_avg = 0

        # shuffle articles and titles (equally)
        c = list(zip(articles, titles))
        random.shuffle(c)
        articles_shuffled, titles_shuffled = zip(*c)

        # split into batches
        article_batches = list(chunks(articles_shuffled, batch_size))
        title_batches = list(chunks(titles_shuffled, batch_size))

        for batch in range(num_batches):
            categories, sequences = random_batch(vocabulary, article_batches[batch], title_batches[batch])

            loss = train(categories, sequences, batch_size, model, optimizer, criterion)

            print_loss_total += loss
            plot_loss_total += loss
            batch_loss_avg += loss
            # calculate number of batches processed
            itr = (epoch-1) * num_batches + batch + 1

            if itr % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                progress, total_runtime = time_since(start, itr / n_iters, total_runtime)
                start = time.time()
                print('%s (%d %d%%) %.4f' % (progress, itr, itr / n_iters * 100, print_loss_avg), flush=True)
                if print_loss_avg < lowest_loss:
                    lowest_loss = print_loss_avg
                    print(" ^ Lowest loss so far", flush=True)

            if itr % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    print("Done with training")


# This can be optimized to instead search for ">>>" to just split on word position
def split_category_and_article(article):
    return article.split(">>>")


def random_batch(vocabulary, articles, titles):
    categories = []
    sequences = []

    batch_size = len(articles)
    for i in range(batch_size):
        category, _ = split_category_and_article(articles[i])
        categories.append(category.strip())
        sequence = indexes_from_sentence(vocabulary, titles[i])
        sequences.append(sequence)

    # Zip into pairs, sort by length (descending), unzip
    # seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    # input_seqs, target_seqs = zip(*seq_pairs)

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    # sequences = Variable(torch.LongTensor(sequences)).transpose(0, 1)
    sequences = Variable(torch.LongTensor(sequences))

    if use_cuda:
        sequences = sequences.cuda()

    return categories, sequences


# TODO: Fix eval

# Eval one batch
# def eval_single_article(categories, sequences, model, criterion):
#     categories_scores = model(sequences)
#
#     category_batch_list = []
#     batch_cat = []
#     for cat in categories[0]:
#         batch_cat.append(int(cat))
#     category_batch_list.append(batch_cat)
#
#     categories = Variable(torch.FloatTensor(category_batch_list))
#     if use_cuda:
#         categories = categories.cuda()
#
#     loss = criterion(categories_scores, categories)
#
#     loss.backward()
#     optimizer.step()
#     return loss.data[0]


if __name__ == '__main__':
    num_articles = 2000
    num_eval = 100

    learning_rate = 0.001
    hidden_size = 128
    dropout_p = 0.5
    num_kernels = 100
    kernel_sizes = [3, 4, 5]
    num_classes = 6

    relative_path = '../data/ntb_processed/ntb_80_6cat.unk'

    articles, titles, vocabulary = preprocess.generate_vocabulary(relative_path, num_articles, True)
    train_articles = articles[0:num_articles]
    train_titles = titles[0:num_articles]
    eval_articles = articles[num_articles:num_articles+num_eval]
    eval_titles = titles[num_articles:num_articles + num_eval]

    model = CNN_Text(vocabulary.n_words, hidden_size, num_classes, num_kernels, kernel_sizes, dropout_p)
    # vocab_size, hidden_size, num_classes, num_kernels, kernel_sizes, dropout_p

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_iters(train_articles, train_titles, vocabulary, model, optimizer)
