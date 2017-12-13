import numpy as np
from tensorboardX import SummaryWriter

from seq2seq_summarization.globals import *
from classifier.train_classifier import get_predictions, calculate_accuracy, create_single_article_category_list
from research.train import split_category_and_article, category_from_string, evaluate


def test_accuracy(config, articles, vocabulary, encoder, decoder, classifier, max_length):
    print("Testing accuracy", flush=True)
    writer = SummaryWriter('../log/test_accuracy1')

    categories_total = []
    categories_scores_total = []

    print("Generating beams", flush=True)
    for i in range(len(articles)):
        print("Evaluating article nr: %d" % i, flush=True)
        category, input_sentence = split_category_and_article(articles[i])
        category = category.strip()
        category_variable = category_from_string(category)
        categories = [category_variable]
        categories_var = Variable(torch.FloatTensor(categories))
        if use_cuda:
            categories_var = categories_var.cuda()

        output_beams = evaluate(config, vocabulary, encoder, decoder, input_sentence, categories_var, max_length)
        top1_beam = output_beams[0]
        top1_sequence_output = top1_beam.decoded_word_sequence
        output_sentence = ' '.join(top1_sequence_output[:-1])

        sequence = indexes_from_sentence(vocabulary, output_sentence)
        sequence = Variable(torch.LongTensor([sequence]))
        if use_cuda:
            sequence = sequence.cuda()

        category = create_single_article_category_list(category)
        categories_total.append(category)
        categories_scores = get_category_scores(sequence, classifier)
        categories_scores_total.append(categories_scores)

    print("Calculating accuracy", flush=True)
    np_gold_truth = np.array(categories_total)

    print(np.shape(np_gold_truth), flush=True)

    np_predicted = get_predictions(categories_scores_total, 0.00)
    print(np.shape(np_predicted), flush=True)

    epoch = 999  # random
    calculate_accuracy(np_gold_truth, np_predicted, writer, epoch)


def get_category_scores(sequence, classifier):
    categories_scores = classifier(sequence, mode='Test')
    return categories_scores.data.cpu().numpy()[0]
