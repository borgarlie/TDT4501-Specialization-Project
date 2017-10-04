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


def train_iters(config, articles, titles, vocabulary, encoder, decoder, max_length,
                encoder_optimizer, decoder_optimizer, writer, start_epoch=1, total_runtime=0):

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

    print("Starting training", flush=True)
    for epoch in range(start_epoch, n_epochs + 1):
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
            input_variable, input_lengths, target_variable, target_lengths = random_batch(batch_size, vocabulary,
                                                                                          article_batches[batch], title_batches[batch], max_length, attention)

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

        # show_plot(plot_losses)


def evaluate(config, vocabulary, encoder, decoder, sentence, max_length):
    attention = config['model']['attention']
    beams = config['evaluate']['beams']
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

    decoder_input = Variable(torch.LongTensor([SOS_token]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoded_words = []

    # TODO: Fix tree search with multiplicative pruning

    # Hard coding the first round in the loop to generate a beam search from the top k candidates of the first word
    if attention:
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, 1)
    else:
        # batch_size = 1 as we do not care about batching during evaluation
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, 1)
    topv, topi = decoder_output.data.topk(beams)
    decoder_input = []
    for i in range(beams):
        decoded_words.append([])
        ni = topi[0][i]
        if ni == EOS_token:
            decoded_words[i].append('<EOS>')
        else:
            decoded_words[i].append(vocabulary.index2word[ni])
        decoder_input1 = Variable(torch.LongTensor([[ni]]))
        decoder_input.append(decoder_input1.cuda() if use_cuda else decoder_input1)

    # looping the beams
    for beam in range(beams):
        if decoded_words[beam][0] != '<EOS>':
            decoded_words[beam] = evaluate_single_beam(vocabulary, decoder, decoded_words[beam], decoder_input[beam],
                                                       decoder_hidden, encoder_outputs, max_length, attention)

    return decoded_words


def evaluate_single_beam(vocabulary, decoder, decoded_words, decoder_input, decoder_hidden, encoder_outputs, max_length, attention=False):
    for di in range(max_length):
        if attention:
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs, 1)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, 1)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(vocabulary.index2word[ni])
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_words


def evaluate_randomly(config, articles, titles, vocabulary, encoder, decoder, max_length):
    for i in range(len(articles)):
        input_sentence = articles[i]
        target_sentence = titles[i]
        print('>', input_sentence, flush=True)
        print('=', target_sentence, flush=True)
        output_words = evaluate(config, vocabulary, encoder, decoder, input_sentence, max_length)
        for beam in output_words:
            if single_char:
                output_sentence = ''.join(beam)  # For single characters
            else:
                output_sentence = ' '.join(beam)  # for words
            print('<', output_sentence, flush=True)
        print('', flush=True)


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


def random_batch(batch_size, vocabulary, articles, titles, max_length, attention=False):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
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

    return input_var, input_lengths, target_var, target_lengths
