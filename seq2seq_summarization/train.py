import os
import random

import torch.nn as nn

from seq2seq_summarization.globals import *


# Train one batch
def train(config, input_variable, input_lengths, target_variable, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    attention = config['model']['attention']
    batch_size = config['train']['batch_size']
    teacher_forcing_ratio = config['train']['teacher_forcing_ratio']

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    max_target_length = max(target_lengths)
    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, None)

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs, batch_size)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, batch_size)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs, batch_size)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, batch_size)
            topv, topi = decoder_output.data.topk(1)
            ni = topi  # next input, batch of top softmax scores
            decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
            loss += criterion(decoder_output, target_variable[di])

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def adjust_learning_rate(optimizer, new_learning_rate):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_learning_rate


def train_iters(config, articles, titles, eval_articles, eval_titles, vocabulary, encoder, decoder, max_length,
                encoder_optimizer, decoder_optimizer, writer, start_epoch=1, total_runtime=0, with_categories=False):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    lowest_loss = 999  # TODO: FIX THIS. save and load

    n_epochs = config['train']['num_epochs']
    batch_size = config['train']['batch_size']
    attention = config['model']['attention']
    print_every = config['log']['print_every']
    plot_every = config['log']['plot_every']

    criterion = nn.NLLLoss()

    num_batches = int(len(articles) / batch_size)
    n_iters = num_batches * n_epochs

    learning_rate = config['train']['learning_rate']
    start_learning_rate_decay_at_epoch = config['train']['decay_epoch']
    decay_frequency = config['train']['decay_frequency']
    counter_since_decay = decay_frequency

    print("Starting training", flush=True)
    for epoch in range(start_epoch, n_epochs + 1):
        print("Starting epoch: %d" % epoch, flush=True)
        batch_loss_avg = 0

        if epoch >= start_learning_rate_decay_at_epoch:
            if counter_since_decay == decay_frequency:
                counter_since_decay = 1
                learning_rate = learning_rate / 2
                adjust_learning_rate(encoder_optimizer, learning_rate)
                adjust_learning_rate(decoder_optimizer, learning_rate)
                print("New learning rate: %.8f" % learning_rate, flush=True)
            else:
                counter_since_decay += 1

        # shuffle articles and titles (equally)
        c = list(zip(articles, titles))
        random.shuffle(c)
        articles_shuffled, titles_shuffled = zip(*c)

        # split into batches
        article_batches = list(chunks(articles_shuffled, batch_size))
        title_batches = list(chunks(titles_shuffled, batch_size))

        for batch in range(num_batches):
            categories, input_variable, input_lengths, target_variable, target_lengths = random_batch(batch_size, vocabulary,
                    article_batches[batch], title_batches[batch], max_length, attention, with_categories)

            loss = train(config, input_variable, input_lengths, target_variable, target_lengths,
                         encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

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
                    # TODO: Consider how we want to save the best model. Only on each epoch or in between?
                    # This current solution will only work for evaluate, and will not work properly if we continue
                    # training on the best model
                    save_state({
                        'epoch': epoch,
                        'runtime': total_runtime,
                        'model_state_encoder': encoder.state_dict(),
                        'model_state_decoder': decoder.state_dict(),
                        'optimizer_state_encoder': encoder_optimizer.state_dict(),
                        'optimizer_state_decoder': decoder_optimizer.state_dict()
                    }, config['experiment_path'] + "/" + config['save']['best_save_file'])

            if itr % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        # log to tensorboard
        writer.add_scalar('loss', batch_loss_avg / num_batches, epoch)

        # save each epoch
        print("Saving model", flush=True)
        itr = epoch * num_batches
        _, total_runtime = time_since(start, itr / n_iters, total_runtime)
        save_state({
            'epoch': epoch,
            'runtime': total_runtime,
            'model_state_encoder': encoder.state_dict(),
            'model_state_decoder': decoder.state_dict(),
            'optimizer_state_encoder': encoder_optimizer.state_dict(),
            'optimizer_state_decoder': decoder_optimizer.state_dict()
        }, config['experiment_path'] + "/" + config['save']['save_file'])

        calculate_loss_on_eval_set(config, vocabulary, encoder, decoder, criterion, writer, epoch, max_length,
                                   eval_articles, eval_titles, with_categories)

        # show_plot(plot_losses)


# Calculate loss on the evaluation set. Does not modify anything.
def calculate_loss_on_eval_set(config, vocabulary, encoder, decoder, criterion, writer, epoch, max_length,
                               eval_articles, eval_titles, with_categories=False):
    attention = config['model']['attention']
    loss = 0
    for i in range(0, len(eval_articles)):
        if with_categories:
            category, article = split_category_and_article(eval_articles[i])
        else:
            article = eval_articles[i]
        title = eval_titles[i]
        if attention:
            input_variable = indexes_from_sentence(vocabulary, article)
            input_variable = pad_seq(input_variable, max_length)
            input_length = max_length
            input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
            input_variable = input_variable.cuda() if use_cuda else input_variable
        else:
            input_variable = variable_from_sentence(vocabulary, article)
            input_length = input_variable.size()[0]

        target_variable = indexes_from_sentence(vocabulary, title)
        target_variable = Variable(torch.LongTensor(target_variable)).unsqueeze(1)
        target_variable = target_variable.cuda() if use_cuda else target_variable

        loss += calculate_loss_on_single_eval_article(attention, encoder, decoder, criterion, input_variable,
                                                      target_variable, input_length)
    loss_avg = loss / len(eval_articles)
    writer.add_scalar('Evaluation loss', loss_avg, epoch)
    print("Evaluation set loss for epoch %d: %.4f" % (epoch, loss_avg), flush=True)


def calculate_loss_on_single_eval_article(attention, encoder, decoder, criterion, input_variable, target_variable,
                                          input_length):
    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)

    decoder_input = Variable(torch.LongTensor([SOS_token]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    for di in range(target_variable.size()[0]):
        if attention:
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                        encoder_outputs, 1)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, 1)
        topv, topi = decoder_output.data.topk(1)
        ni = topi  # next input, batch of top softmax scores
        decoder_input = Variable(torch.cuda.LongTensor(ni)) if use_cuda else Variable(torch.LongTensor(ni))
        loss += criterion(decoder_output, target_variable[di])

    return loss.data[0]


def evaluate_randomly(config, articles, titles, vocabulary, encoder, decoder, max_length, with_categories=False):
    for i in range(len(articles)):
        if with_categories:
            category, input_sentence = split_category_and_article(articles[i])
        else:
            input_sentence = articles[i]
        target_sentence = titles[i]
        print('>', input_sentence, flush=True)
        print('=', target_sentence, flush=True)
        output_beams = evaluate(config, vocabulary, encoder, decoder, input_sentence, max_length)
        for beam in output_beams:
            output_words = beam.decoded_word_sequence
            if single_char:
                output_sentence = ''.join(output_words)  # For single characters
            else:
                output_sentence = ' '.join(output_words)  # for words
            print('<', str(beam.get_avg_score()), output_sentence, flush=True)
        print('', flush=True)


def evaluate(config, vocabulary, encoder, decoder, sentence, max_length):
    attention = config['model']['attention']
    if attention:
        input_variable = indexes_from_sentence(vocabulary, sentence)
        input_variable = pad_seq(input_variable, max_length)
        input_length = max_length
        input_variable = Variable(torch.LongTensor(input_variable)).unsqueeze(1)
        input_variable = input_variable.cuda() if use_cuda else input_variable
    else:
        input_variable = variable_from_sentence(vocabulary, sentence)
        input_length = input_variable.size()[0]

    encoder_outputs, encoder_hidden = encoder(input_variable, [input_length], None)

    expansions = config['evaluate']['expansions']
    keep_beams = config['evaluate']['keep_beams']
    return_beams = config['evaluate']['return_beams']

    # first decoder beam. input_hidden = encoder_hidden
    beams = [Beam([], [], SOS_token, encoder_hidden)]
    for i in range(max_length):
        beams = expand_and_prune_beams(vocabulary, beams, encoder_outputs, decoder, attention, expansions, keep_beams)

    return prune_beams(beams, return_beams)


def expand_and_prune_beams(vocabulary, beams, encoder_outputs, decoder, attention=False, expansions=5, keep_beams=10):
    generated_beams = []
    for i in range(len(beams)):
        generated_beams += beams[i].expand(vocabulary, encoder_outputs, decoder, attention, expansions)
    return prune_beams(generated_beams, keep_beams)


# Takes in a set of beams and returns the best scoring beams up until num_keep_beams
def prune_beams(beams, num_keep_beams):
    return sorted(beams, reverse=True)[:num_keep_beams]


class Beam:
    def __init__(self, decoded_word_sequence, scores, input_token, input_hidden):
        self.decoded_word_sequence = decoded_word_sequence
        self.scores = scores  # This is a list of log(output from softmax) for each word in the sequence
        self.input_token = input_token
        self.input_hidden = input_hidden

    def get_avg_score(self):
        if len(self.scores) == 0:
            return 0.0
        return sum(self.scores) / len(self.scores)

    def __lt__(self, other):
        return self.get_avg_score().__lt__(other.get_avg_score())

    def generate_expanded_beams(self, vocabulary, topv, topi, decoder_hidden, expansions=5):
        for i in range(expansions):
            next_word = topi[0][i]
            decoded_words = list(self.decoded_word_sequence) + [vocabulary.index2word[next_word]]
            # Using log(score) to be able to sum instead of multiply,
            # so that we are able to take the average based on number of tokens in the sequence
            # next_score = numpy.log2(topv[0][i]) - already using log Softmax
            next_score = topv[0][i]
            new_scores = list(self.scores) + [next_score]
            yield Beam(decoded_words, new_scores, next_word, decoder_hidden)

    # return list of expanded beams. return self if current beam is at end of sentence
    def expand(self, vocabulary, encoder_outputs, decoder, attention=False, expansions=5):
        if self.input_token == EOS_token or self.input_token == PAD_token:
            return list([self])
        # expand beam
        decoder_input = Variable(torch.LongTensor([self.input_token]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
        if attention:
            decoder_output, decoder_hidden, _ = decoder(decoder_input, self.input_hidden, encoder_outputs, 1)
        else:
            # batch_size = 1 as we do not care about batching during evaluation
            decoder_output, decoder_hidden = decoder(decoder_input, self.input_hidden, 1)
        topv, topi = decoder_output.data.topk(expansions)
        return list(self.generate_expanded_beams(vocabulary, topv, topi, decoder_hidden, expansions))

    def __repr__(self):
        return str(self.get_avg_score())

    def __str__(self):
        return self.__repr__()


# def evaluate_single_beam(vocabulary, decoder, decoded_words, decoder_input, decoder_hidden, encoder_outputs, max_length, attention=False):
#     for di in range(max_length):
#         if attention:
#             decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, 1)
#         else:
#             decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, 1)
#         topv, topi = decoder_output.data.topk(1)
#         ni = topi[0][0]
#         if ni == EOS_token:
#             decoded_words.append('<EOS>')
#             break
#         else:
#             decoded_words.append(vocabulary.index2word[ni])
#         decoder_input = Variable(torch.LongTensor([[ni]]))
#         decoder_input = decoder_input.cuda() if use_cuda else decoder_input
#
#     return decoded_words


def save_state(state, filename):
    torch.save(state, filename)


def load_state(filename):
    if os.path.isfile(filename):
        state = torch.load(filename)
        return (state['epoch'], state['runtime'],
                state['model_state_encoder'], state['model_state_decoder'],
                state['optimizer_state_encoder'], state['optimizer_state_decoder'])
    else:
        raise FileNotFoundError


# This can be optimized to instead search for ">>>" to just split on word position
def split_category_and_article(article):
    return article.split(">>>")


def random_batch(batch_size, vocabulary, articles, titles, max_length, attention=False, with_categories=False):
    input_seqs = []
    categories = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        if with_categories:
            category, article = split_category_and_article(articles[i])
            categories.append(category.strip())
            input_variable = indexes_from_sentence(vocabulary, article.strip())  # is ".strip" necessary?
        else:
            input_variable = indexes_from_sentence(vocabulary, articles[i])
        target_variable = indexes_from_sentence(vocabulary, titles[i])
        input_seqs.append(input_variable)
        target_seqs.append(target_variable)

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    if attention:
        input_lengths = [max_length for s in input_seqs]
        input_padded = [pad_seq(s, max_length) for s in input_seqs]
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [pad_seq(s, max_length) for s in target_seqs]
    else:
        input_lengths = [len(s) for s in input_seqs]
        input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]
        target_lengths = [len(s) for s in target_seqs]
        target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)

    if use_cuda:
        input_var = input_var.cuda()
        target_var = target_var.cuda()

    return categories, input_var, input_lengths, target_var, target_lengths
