import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch


def read_file(path, name):
    file = open(path + name, encoding='utf-8').read().strip().split('\n')
    return file


# This can be optimized to instead search for ">>>" to just split on word position
def split_category_and_article(article):
    return article.split(">>>")[1].strip()


def return_category_name(category):
    if category == 0:
        return "Sport"
    elif category == 1:
        return "Økonomi og næringsliv"
    elif category == 2:
        return "Politikk"
    elif category == 3:
        return "Kriminalitet og rettsvesen"
    elif category == 4:
        return "Ulykker og naturkatastrofer"
    else:
        return "wtf"


if __name__ == '__main__':
    relative_path = 'research_experiments/attention/'
    num_tests = 5

    attentions_weights = torch.load(relative_path + 'attention_weights')
    articles = read_file(relative_path, 'test_articles.txt')
    titles = read_file(relative_path, 'test_titles.txt')

    fig = plt.figure()

    for i in range(num_tests):
        test_article = split_category_and_article(articles[i])

        input_sentence = test_article
        output_words = titles[i]
        attentions = attentions_weights[i]

        # Set up figure with color bar
        ax = fig.add_subplot(5, 2, i+1)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)

        # Set up axes
        ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
        ax.set_yticklabels([''] + output_words.split(' '))
        ax.set_title(return_category_name(i))

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
