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
          encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, attention=False, batch_size=1,
          teacher_forcing_ratio=0.5):

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


# Train for a number of epochs
def train_iters(articles, titles, vocabulary, encoder, decoder, n_epochs, max_length,
                encoder_optimizer, decoder_optimizer, save_file, best_model_save_file, save_every=-1,
                start_epoch=1, total_runtime=0, print_every=1000, plot_every=100, attention=False, batch_size=1,
                teacher_forcing_ratio=0.5):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    lowest_loss = 999

    criterion = nn.NLLLoss()

    num_batches = int(len(articles) / batch_size)
    n_iters = num_batches * n_epochs

    print("Starting training", flush=True)
    for epoch in range(start_epoch, n_epochs + 1):
        print("Starting epoch: %d" % epoch, flush=True)

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

            loss = train(input_variable, input_lengths, target_variable, target_lengths,
                         encoder, decoder, encoder_optimizer, decoder_optimizer,
                         criterion, attention=attention, batch_size=batch_size,
                         teacher_forcing_ratio=teacher_forcing_ratio)
            print_loss_total += loss
            plot_loss_total += loss

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
                    if save_every > 0:
                        save_state({
                            'epoch': epoch,
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

        # save each epoch
        print("Saving model", flush=True)
        itr = epoch * num_batches
        _, total_runtime = time_since(start, itr / n_iters, total_runtime)
        save_state({
            'epoch': epoch,
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
            decoded_words[beam] = evaluate_single_beam(decoder, decoded_words[beam], decoder_input[beam],
                                                       decoder_hidden, encoder_outputs, max_length, attention)

    return decoded_words


def evaluate_single_beam(decoder, decoded_words, decoder_input, decoder_hidden, encoder_outputs, max_length, attention=False):
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


if __name__ == '__main__':

    print(use_cuda, flush=True)
    if use_cuda and len(sys.argv) == 2:
        torch.cuda.set_device(int(sys.argv[1]))
        print("Changed to GPU: %s" % sys.argv[1], flush=True)

    # Train and evaluate parameters
    relative_path = '../data/articles2_nor/25to100'
    num_articles = 150  # -1 means to take the maximum from the provided source
    num_evaluate = 10
    num_epochs = 10
    start_epoch = 1
    total_runtime = 0
    beams = 5
    batch_size = 16

    # Model parameters
    attention = False
    hidden_size = 128
    n_layers = 1
    dropout_p = 0.1
    learning_rate = 0.01
    teacher_forcing_ratio = 0.5

    print("Hidden size: %d" % hidden_size, flush=True)
    print("Attention: " + str(attention), flush=True)
    print("n-layers: %d" % n_layers, flush=True)
    print("Batch size: %d" % batch_size, flush=True)
    print("Teacher forcing: %0.1f" % teacher_forcing_ratio, flush=True)

    save_file = '../saved_models/testing/test3_2.pth.tar'
    best_model_save_file = '../saved_models/testing/test3_2_best.pth.tar'

    # When loading a model - the hyper parameters defined here are ignored as they are set in the model
    load_model = False
    load_file = '../saved_models/testing/test3_2.pth.tar'

    save_every = 1000  # -1 means never to safe. Must be a multiple of print_every
    print_every = 10  # Also used for plotting

    if save_every % print_every != 0:
        raise ValueError("Print_every must be a multiple of save_every, but is currently %d and %d"
                         % (save_every, print_every))

    pre = preprocess_single_char if single_char else preprocess
    articles, titles, vocabulary = pre.generate_vocabulary(relative_path, num_articles)

    total_articles = len(articles)
    train_articles_length = total_articles - num_evaluate

    # Append remainder to evaluate set so that the training set has exactly a multiple of batch size
    num_evaluate += train_articles_length % batch_size
    train_length = total_articles - num_evaluate
    test_length = num_evaluate
    print("Train length = %d" % train_length)
    print("Test length = %d" % test_length)

    train_articles = articles[0:train_length]
    train_titles = titles[0:train_length]
    test_articles = articles[train_length:train_length + test_length]
    test_titles = titles[train_length:train_length + test_length]

    encoder1 = EncoderRNN(vocabulary.n_words, hidden_size, n_layers=n_layers, batch_size=batch_size)

    max_length = max(len(article.split(' ')) for article in articles) + 1

    if attention:
        decoder1 = AttnDecoderRNN(hidden_size, vocabulary.n_words, max_length=max_length, n_layers=n_layers,
                                  dropout_p=dropout_p, batch_size=batch_size)
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
            print("Resuming training from iteration: %d" % start_iter, flush=True)
        except FileNotFoundError:
            print("No file found: exiting", flush=True)
            exit()

    train_iters(train_articles, train_titles, vocabulary, encoder1, decoder1, num_epochs, max_length,
                encoder_optimizer, decoder_optimizer, save_file, best_model_save_file,
                save_every=save_every, start_epoch=start_epoch, total_runtime=total_runtime,
                print_every=print_every, plot_every=print_every, attention=attention, batch_size=batch_size,
                teacher_forcing_ratio=teacher_forcing_ratio)
    # When saving the best model, the average is counted over "print every", to not have this randomly be very low,
    # we need to have it large enough, I.e. at least 100 ++

    evaluate_randomly(test_articles, test_titles, vocabulary, encoder1, decoder1, max_length=max_length,
                      attention=attention, beams=beams)
