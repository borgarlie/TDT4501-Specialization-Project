import random

import os
from torch import optim
import preprocess_single_char as preprocess_single_char
import preprocess as preprocess
from encoder import *
from decoder import *
from globals import *

current_iteration = 0

######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length, attention=False):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            if attention:
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                            encoder_outputs)
            else:
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def train_iters(articles, titles, vocabulary, encoder, decoder, n_iters, max_length,
                encoder_optimizer, decoder_optimizer, save_file, best_model_save_file, save_every=-1,
                start_iter=1, total_runtime=0, print_every=1000, plot_every=100, attention=False):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    lowest_loss = 999

    criterion = nn.NLLLoss()

    print("Starting training", flush=True)

    for itr in range(start_iter, n_iters + 1):
        random_number = random.randint(0, len(articles) - 1)
        input_variable, target_variable = variables_from_pair(articles[random_number], titles[random_number],
                                                              vocabulary)

        loss = train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer,
                     criterion, max_length=max_length, attention=attention)
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

    show_plot(plot_losses)


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.


def evaluate(vocabulary, encoder, decoder, sentence, max_length, attention=False, beams=3):
    input_variable = variable_from_sentence(vocabulary, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoded_words = []

    # Hard coding the first round in the loop to generate a beam search from the top k candidates of the first word
    if attention:
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
    else:
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.data.topk(beams)  # standard was 1
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


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:


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


if __name__ == '__main__':

    print(use_cuda, flush=True)

    # Train and evaluate parameters
    relative_path = '../data/articles2_nor/25to100'
    num_articles = -1  # -1 means to take the maximum from the provided source
    num_evaluate = 300
    iterations = 150000
    start_iter = 1
    total_runtime = 0
    beams = 5

    # Model parameters
    attention = False
    hidden_size = 128
    max_length = 1 + 100  # WOWSKI ... 1029
    n_layers = 1
    dropout_p = 0.1
    learning_rate = 0.01
    # TODO: Add teacher forcing here instead of in globals

    save_file = '../saved_models/testing/test1.pth.tar'
    best_model_save_file = '../saved_models/testing/test1_best.pth.tar'

    # When loading a model - the hyper parameters defined here are ignored as they are set in the model
    load_model = False
    load_file = '../saved_models/testing/seq2seq_char_hidden128_layer1_len80.pth.tar'

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

    encoder1 = EncoderRNN(vocabulary.n_words, hidden_size, n_layers=n_layers)

    if attention:
        decoder1 = AttnDecoderRNN(hidden_size, vocabulary.n_words, max_length=max_length, n_layers=n_layers,
                                  dropout_p=dropout_p)
    else:
        decoder1 = DecoderRNN(hidden_size, vocabulary.n_words, n_layers=n_layers)

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

    train_iters(train_articles, train_titles, vocabulary, encoder1, decoder1, iterations, max_length,
                encoder_optimizer, decoder_optimizer, save_file, best_model_save_file,
                save_every=save_every, start_iter=start_iter, total_runtime=total_runtime,
                print_every=print_every, plot_every=print_every, attention=attention)
    # When saving the best model, the average is counted over "print every", to not have this randomly be very low,
    # we need to have it large enough, I.e. at least 100 ++

    evaluate_randomly(test_articles, test_titles, vocabulary, encoder1, decoder1, max_length=max_length,
                      attention=attention, beams=beams)
