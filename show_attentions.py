import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with color bar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words.split(' '))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def read_file(path, name):
    file = open(path + name, encoding='utf-8').read().strip().split('\n')
    return file


# This can be optimized to instead search for ">>>" to just split on word position
def split_category_and_article(article):
    return article.split(">>>")[1].strip()


if __name__ == '__main__':
    relative_path = 'data/attention_data/'
    num_tests = 5

    attentions_weights = torch.load(relative_path + 'attention_weights')
    articles = read_file(relative_path, 'test_articles.txt')
    titles = read_file(relative_path, 'test_titles.txt')

    for i in range(num_tests):
        test_article = split_category_and_article(articles[i])
        show_attention(test_article, titles[i], attentions_weights[i])
