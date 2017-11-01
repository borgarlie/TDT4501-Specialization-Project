import json
import sys
sys.path.append('..')  # ugly dirtyfix for imports to work

import random

import torch.nn as nn
from torch import optim

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

from classifier.cnn_classifier import CNN_Text as CNN_Text
from seq2seq_summarization.globals import *
from seq2seq_summarization import preprocess as preprocess
from tensorboardX import SummaryWriter


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


def train_iters(articles, titles, vocabulary, model, optimizer, eval_articles, eval_titles, writer, batch_size
                , n_epochs):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    print_every = 500
    plot_every = 500

    criterion = nn.BCEWithLogitsLoss()

    num_batches = int(len(articles) / batch_size)
    n_iters = num_batches * n_epochs

    lowest_loss = 999
    total_runtime = 0.0

    # Evaluate once before starting to train
    evaluate(eval_articles, eval_titles, vocabulary, model, writer, 1, 0)

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

        batch_loss_avg /= num_batches
            # evaluate epoch on test set
        evaluate(eval_articles, eval_titles, vocabulary, model, writer, batch_loss_avg, epoch)

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

    seq_lengths = [len(s) for s in sequences]
    seq_padded = [pad_seq(s, max(seq_lengths)) for s in sequences]

    sequences = Variable(torch.LongTensor(seq_padded))

    if use_cuda:
        sequences = sequences.cuda()

    return categories, sequences


def calculate_accuracy(gold_truth, predictions, writer, epoch):
    # calculate total accuracy
    accuracy = accuracy_score(gold_truth, predictions)
    print("Total Accuracy: %0.4f" % accuracy, flush=True)

    # Accuracy dictionary for tensorboard
    accuracy_dict = {'Total accuracy': accuracy}

    # calculate accuracy for single categories
    num_categories = len(gold_truth[0])
    for i in range(0, num_categories):
        gold_truth_i = []
        predictions_i = []
        for k in range(0, len(gold_truth)):
            gold_truth_i.append([gold_truth[k][i]])
            predictions_i.append([predictions[k][i]])
        gold_truth_i = np.array(gold_truth_i)
        predictions_i = np.array(predictions_i)
        accuracy_i = accuracy_score(gold_truth_i, predictions_i)
        print("Accuracy for category %d: %0.4f" % (i, accuracy_i), flush=True)
        accuracy_dict.update({'Accuracy for category %d' % i: accuracy_i})
    writer.add_scalars('Accuracy', accuracy_dict, epoch)


def calculate_f1(gold_truth, predictions, writer, epoch):
    precision, recall, f1_score, support = precision_recall_fscore_support(gold_truth, predictions)
    writer_dict = {}
    num_categories = len(gold_truth[0])
    for i in range(0, num_categories):
        writer_dict["Precision %d" % i] = precision[i]
        writer_dict["Recall %d" % i] = recall[i]
        writer_dict["f1_score %d" % i] = f1_score[i]
        writer_dict["support %d" % i] = float(support[i])
    print(writer_dict, flush=True)
    print(json.dumps(writer_dict, indent=2), flush=True)
    writer.add_scalars('Precision_recall_f1', writer_dict, epoch)


def get_predictions(model_scores, min_score):
    predicted = []
    for example in range(0, len(model_scores)):
        example_predictions = []
        for score in range(0, len(model_scores[example])):
            if model_scores[example][score] > min_score:
                example_predictions.append(1)
            else:
                example_predictions.append(0)
        predicted.append(example_predictions)
    return np.array(predicted)


def eval_single_article(category, sequence, model, criterion):
    categories_scores = model(sequence, mode='Test')

    categories = Variable(torch.FloatTensor([category]))
    if use_cuda:
        categories = categories.cuda()

    loss = criterion(categories_scores, categories)
    return loss.data[0], categories_scores.data.cpu().numpy()[0]


def create_single_article_category_list(category_string):
    categories = []
    for cat in category_string:
        categories.append(int(cat))
    return categories


def evaluate(articles, titles, vocabulary, model, writer, train_loss, epoch):
    print("Evaluating", flush=True)
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0
    categories_total = []
    categories_scores_total = []

    for i in range(len(articles)):
        category, _ = split_category_and_article(articles[i])
        category = category.strip()  # is .strip() needed?
        category = create_single_article_category_list(category)
        categories_total.append(category)
        sequence = indexes_from_sentence(vocabulary, titles[i])
        sequence = Variable(torch.LongTensor([sequence]))
        if use_cuda:
            sequence = sequence.cuda()
        loss, categories_score = eval_single_article(category, sequence, model, criterion)
        categories_scores_total.append(categories_score)
        total_loss += loss
    avg_loss = total_loss / len(articles)
    print("Avg evaluation loss: %0.4f" % avg_loss, flush=True)

    # Making a loss dictionary for tensorboard
    loss_dict = {'train_loss': train_loss, 'eval_loss': avg_loss}
    writer.add_scalars('Loss', loss_dict, epoch)

    np_gold_truth = np.array(categories_total)

    print("Calculating accuracy on 0.00 confidence", flush=True)
    np_predicted = get_predictions(categories_scores_total, 0.00)
    calculate_accuracy(np_gold_truth, np_predicted, writer, epoch)
    print("Calculating f1 score", flush=True)
    calculate_f1(np_gold_truth, np_predicted, writer, epoch)


if __name__ == '__main__':

    if use_cuda:
        if len(sys.argv) < 2:
            print("Expected 1 arguments: [0] = GPU (0 or 1)", flush=True)
            exit()
        torch.cuda.set_device(int(sys.argv[1]))
        print("Using GPU: %s" % sys.argv[1], flush=True)

    writer = SummaryWriter('../log/test_classifier')

    num_articles = -1
    num_eval = 6500

    learning_rate = 0.001
    hidden_size = 128
    dropout_p = 0.5
    num_kernels = 100
    kernel_sizes = [3, 4, 5]
    num_classes = 5

    n_epochs = 5
    batch_size = 16

    print("Using cuda: " + str(use_cuda), flush=True)

    relative_path = '../data/ntb_processed/ntb_80_5cat.unk'

    articles, titles, vocabulary = preprocess.generate_vocabulary(relative_path, num_articles, True)

    # shuffle articles and titles (equally)
    # c = list(zip(articles, titles))
    # random.shuffle(c)
    # articles_shuffled, titles_shuffled = zip(*c)

    train_articles = articles[0:num_articles-num_eval]
    train_titles = titles[0:num_articles-num_eval]
    eval_articles = articles[num_articles-num_eval:num_articles]
    eval_titles = titles[num_articles-num_eval:num_articles]

    model = CNN_Text(vocabulary.n_words, hidden_size, num_classes, num_kernels, kernel_sizes, dropout_p)
    if use_cuda:
        model = model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_iters(train_articles, train_titles, vocabulary, model, optimizer, eval_articles, eval_titles,
                writer, batch_size, n_epochs)

    writer.close()
