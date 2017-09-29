import random
import os
from torch import optim
import preprocess_single_char as preprocess_single_char
import preprocess as preprocess
from encoder import *
from decoder import *
from globals import *
import sys


# Train one batch
def train(input_variable, input_lengths, target_variable, target_lengths,
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attention=False, batch_size=1):

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    max_target_length = max(target_lengths)
    loss = 0
    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, None)

    decoder_input = Variable(torch.LongTensor([SOS_token] * batch_size))  # does it need an outer [] ?
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(max_target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, batch_size)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(max_target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
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


# Train for a number of epochs
def train_iters(articles, titles, vocabulary, encoder, decoder, n_iters,
                encoder_optimizer, decoder_optimizer, save_file, best_model_save_file, save_every=-1,
                start_iter=1, total_runtime=0, print_every=1000, plot_every=100, attention=False, batch_size=1):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    lowest_loss = 999

    criterion = nn.NLLLoss()

    print("Starting training", flush=True)

    # TODO: Fix shuffle instead of random (so we take every input as many times)
    # TODO: Should we fix such that the input_num_articles % batch_size == 0, and put the remaining in evaluation ?
    # TODO: Rename iterations to epochs
    for itr in range(start_iter, n_iters + 1):
        # random_number = random.randint(0, len(articles) - 1)
        # input_variable, target_variable = variables_from_pair(articles[random_number], titles[random_number],
        #                                                       vocabulary)
        input_variable, input_lengths, target_variable, target_lengths = random_batch(batch_size, vocabulary, articles,
                                                                                      titles)

        loss = train(input_variable, input_lengths, target_variable, target_lengths,
                     encoder, decoder, encoder_optimizer, decoder_optimizer,
                     criterion, attention=attention, batch_size=batch_size)
        print_loss_total += loss
        plot_loss_total += loss

        if itr % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            progress, total_runtime = time_since(start, itr / n_iters, total_runtime)
            start = time.time()
            print('%s (%d %d%%) %.4f' % (progress, itr, itr / n_iters * 100, print_loss_avg), flush=True)
            if print_loss_avg < lowest_loss:
                lowest_loss = print_loss_avg
                print(" ^ Lowest loss so far", flush=True)
                if save_every > 0:
                    save_state({
                        'iteration': itr + 1,
                        'runtime': total_runtime,
                        'attention': attention,
                        'max_length': max_length,
                        'model_state_encoder': encoder1.state_dict(),
                        'model_state_decoder': decoder1.state_dict(),
                        'optimizer_state_encoder': encoder_optimizer.state_dict(),
                        'optimizer_state_decoder': decoder_optimizer.state_dict()
                    }, best_model_save_file)

        if itr % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if save_every > 0 and itr % save_every == 0:
            save_state({
                'iteration': itr+1,
                'runtime': total_runtime,
                'attention': attention,
                'max_length': max_length,
                'model_state_encoder': encoder1.state_dict(),
                'model_state_decoder': decoder1.state_dict(),
                'optimizer_state_encoder': encoder_optimizer.state_dict(),
                'optimizer_state_decoder': decoder_optimizer.state_dict()
            }, save_file)

    # show_plot(plot_losses)


def evaluate(vocabulary, encoder, decoder, sentence, max_length, attention=False, beams=3):
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
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
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
            decoded_words[beam] = evaluate_single_beam(decoder, decoded_words[beam], decoder_input[beam],
                                                       decoder_hidden, encoder_outputs, max_length, attention)

    return decoded_words


def evaluate_single_beam(decoder, decoded_words, decoder_input, decoder_hidden, encoder_outputs, max_length, attention=False):
    for di in range(max_length):
        if attention:
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        else:
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
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


def evaluate_randomly(articles, titles, vocabulary, encoder, decoder, max_length, attention=False, beams=3):
    for i in range(len(articles)):
        input_sentence = articles[i]
        target_sentence = titles[i]
        print('>', input_sentence, flush=True)
        print('=', target_sentence, flush=True)
        output_words = evaluate(vocabulary, encoder, decoder, input_sentence, max_length=max_length,
                                attention=attention, beams=beams)
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
        return (state['iteration'], state['runtime'], state['attention'], state['max_length'],
                state['model_state_encoder'], state['model_state_decoder'],
                state['optimizer_state_encoder'], state['optimizer_state_decoder'])
    else:
        raise FileNotFoundError


# TODO: FIx so its not as random.. just shuffled
def random_batch(batch_size, vocabulary, articles, titles):
    input_seqs = []
    target_seqs = []

    # Choose random pairs
    for i in range(batch_size):
        random_number = random.randint(0, len(articles) - 1)
        input_variable = indexes_from_sentence(vocabulary, articles[random_number])
        target_variable = indexes_from_sentence(vocabulary, titles[random_number])
        input_seqs.append(input_variable)
        target_seqs.append(target_variable)

    # Zip into pairs, sort by length (descending), unzip
    seq_pairs = sorted(zip(input_seqs, target_seqs), key=lambda p: len(p[0]), reverse=True)
    input_seqs, target_seqs = zip(*seq_pairs)

    # For input and target sequences, get array of lengths and pad with 0s to max length
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


if __name__ == '__main__':

    print(use_cuda, flush=True)
    if use_cuda and len(sys.argv) == 2:
        torch.cuda.set_device(int(sys.argv[1]))
        print("Changed to GPU: %s" % sys.argv[1], flush=True)

    # Train and evaluate parameters
    relative_path = '../data/articles2_nor/25to100'
    num_articles = -1  # -1 means to take the maximum from the provided source
    num_evaluate = 10
    iterations = 100
    start_iter = 1
    total_runtime = 0
    beams = 3
    batch_size = 4

    # Model parameters
    attention = False
    hidden_size = 128
    max_length = 1 + 100  # TODO: This should probably be removed fully
    n_layers = 1
    dropout_p = 0.1
    learning_rate = 0.01
    # TODO: Add teacher forcing here instead of in globals
    # TODO: Fix attention decoder

    save_file = '../saved_models/testing/test3.pth.tar'
    best_model_save_file = '../saved_models/testing/test3_best.pth.tar'

    # When loading a model - the hyper parameters defined here are ignored as they are set in the model
    load_model = False
    load_file = '../saved_models/testing/test3.pth.tar'

    save_every = 1000  # -1 means never to safe. Must be a multiple of print_every
    print_every = 10  # Also used for plotting

    if save_every % print_every != 0:
        raise ValueError("Print_every must be a multiple of save_every, but is currently %d and %d"
                         % (save_every, print_every))

    pre = preprocess_single_char if single_char else preprocess
    articles, titles, vocabulary = pre.generate_vocabulary(relative_path, num_articles)

    train_length = num_articles - num_evaluate
    test_length = num_evaluate

    train_articles = articles[0:train_length]
    train_titles = titles[0:train_length]
    test_articles = articles[train_length:train_length + test_length]
    test_titles = titles[train_length:train_length + test_length]

    encoder1 = EncoderRNN(vocabulary.n_words, hidden_size, n_layers=n_layers, batch_size=batch_size)

    if attention:
        decoder1 = AttnDecoderRNN(hidden_size, vocabulary.n_words, max_length=max_length, n_layers=n_layers,
                                  dropout_p=dropout_p)
    else:
        decoder1 = DecoderRNN(hidden_size, vocabulary.n_words, n_layers=n_layers, batch_size=batch_size)

    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder1.parameters(), lr=learning_rate)

    if load_model:
        try:
            (start_iter, total_runtime, attention, max_length, model_state_encoder,
             model_state_decoder, optimizer_state_encoder, optimizer_state_decoder) = load_state(load_file)
            encoder1.load_state_dict(model_state_encoder)
            decoder1.load_state_dict(model_state_decoder)
            encoder_optimizer.load_state_dict(optimizer_state_encoder)
            decoder_optimizer.load_state_dict(optimizer_state_decoder)
            print("Resuming training from iteration: %d" % start_iter)
        except FileNotFoundError:
            print("No file found: exiting")
            exit()

    train_iters(train_articles, train_titles, vocabulary, encoder1, decoder1, iterations,
                encoder_optimizer, decoder_optimizer, save_file, best_model_save_file,
                save_every=save_every, start_iter=start_iter, total_runtime=total_runtime,
                print_every=print_every, plot_every=print_every, attention=attention, batch_size=batch_size)
    # When saving the best model, the average is counted over "print every", to not have this randomly be very low,
    # we need to have it large enough, I.e. at least 100 ++

    evaluate_randomly(test_articles, test_titles, vocabulary, encoder1, decoder1, max_length=max_length,
                      attention=attention, beams=beams)
