import json
import sys
sys.path.append('..')  # ugly dirtyfix for imports to work


from seq2seq_summarization import preprocess as preprocess
from seq2seq_summarization import preprocess_single_char as preprocess_single_char
from seq2seq_summarization.decoder import *
from seq2seq_summarization.encoder import *
from seq2seq_summarization.globals import *
from seq2seq_summarization.train import *
from torch import optim
from tensorboardX import SummaryWriter


if __name__ == '__main__':

    if use_cuda:
        if len(sys.argv) < 3:
            print("Expected 2 arguments: [0] = experiment path (e.g. test_experiment1), [1] = GPU (0 or 1)", flush=True)
            exit()
        torch.cuda.set_device(int(sys.argv[2]))
        print("Using GPU: %s" % sys.argv[2], flush=True)
    else:
        if len(sys.argv) < 2:
            print("Expected 1 argument: [0] = experiment path (e.g. test_experiment1)", flush=True)
            exit()

    experiment_path = sys.argv[1]
    config_file_path = experiment_path + "/config.json"
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    config['experiment_path'] = experiment_path
    print(json.dumps(config, indent=2), flush=True)

    writer = SummaryWriter(config['tensorboard']['log_path'])
    relative_path = config['train']['dataset']
    num_articles = config['train']['num_articles']
    num_evaluate = config['train']['num_evaluate']
    num_throw = config['train']['throw']
    batch_size = config['train']['batch_size']
    learning_rate = config['train']['learning_rate']

    attention = config['model']['attention']
    hidden_size = config['model']['hidden_size']
    n_layers = config['model']['n_layers']
    dropout_p = config['model']['dropout_p']

    load_model = config['train']['load']
    load_file = experiment_path + "/" + config['train']['load_file']

    pre = preprocess_single_char if single_char else preprocess
    articles, titles, vocabulary = pre.generate_vocabulary(relative_path, num_articles)

    total_articles = len(articles) - num_throw
    train_articles_length = total_articles - num_evaluate

    # Append remainder to evaluate set so that the training set has exactly a multiple of batch size
    num_evaluate += train_articles_length % batch_size
    train_length = total_articles - num_evaluate
    test_length = num_evaluate
    print("Train length = %d" % train_length, flush=True)
    print("Throw length = %d" % num_throw, flush=True)
    print("Test length = %d" % test_length, flush=True)

    train_articles = articles[0:train_length]
    train_titles = titles[0:train_length]
    print("Range train: %d - %d" % (0, train_length), flush=True)

    train_length = train_length + num_throw  # compensate for thrown away articles
    test_articles = articles[train_length:train_length + test_length]
    test_titles = titles[train_length:train_length + test_length]

    print("Range test: %d - %d" % (train_length, train_length+test_length), flush=True)

    encoder = EncoderRNN(vocabulary.n_words, hidden_size, n_layers=n_layers, batch_size=batch_size)

    max_length = max(len(article.split(' ')) for article in articles) + 1

    if attention:
        decoder = AttnDecoderRNN(hidden_size, vocabulary.n_words, max_length=max_length, n_layers=n_layers,
                                 dropout_p=dropout_p, batch_size=batch_size)
    else:
        decoder = DecoderRNN(hidden_size, vocabulary.n_words, n_layers=n_layers, batch_size=batch_size)

    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    total_runtime = 0
    start_epoch = 1
    if load_model:
        try:
            (start_epoch, total_runtime, model_state_encoder, model_state_decoder,
             optimizer_state_encoder, optimizer_state_decoder) = load_state(load_file)
            encoder.load_state_dict(model_state_encoder)
            decoder.load_state_dict(model_state_decoder)
            encoder_optimizer.load_state_dict(optimizer_state_encoder)
            decoder_optimizer.load_state_dict(optimizer_state_decoder)
            print("Resuming training from epoch: %d" % start_epoch, flush=True)
        except FileNotFoundError:
            print("No file found: exiting", flush=True)
            exit()

    train_iters(config, train_articles, train_titles, vocabulary, encoder, decoder, max_length,
                encoder_optimizer, decoder_optimizer, writer, start_epoch=start_epoch, total_runtime=total_runtime)

    evaluate_randomly(config, test_articles, test_titles, vocabulary, encoder, decoder, max_length=max_length)
