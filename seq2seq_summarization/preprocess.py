from __future__ import unicode_literals, print_function, division
from io import open


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.


class Vocabulary:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def generate_vocabulary():
    print("Reading lines...")
    article = open('../data/articles1/valid.article.filter.txt', encoding='utf-8').read().strip().split('\n')
    title = open('../data/articles1/valid.title.filter.txt', encoding='utf-8').read().strip().split('\n')
    print("Read %s articles" % len(article))
    print("Read %s title" % len(title))

    vocabulary = Vocabulary()

    longest_sentence = 0

    max_size = 100

    print("Counting words...")
    # for sentence in article:
    for sentence in range(0, max_size):
        vocabulary.add_sentence(article[sentence])
        if len(article[sentence].split(' ')) > longest_sentence:
            longest_sentence = len(article[sentence].split(' '))
            # print(sentence)
    for sentence in range(0, max_size):
        vocabulary.add_sentence(title[sentence])
        if len(title[sentence].split(' ')) > longest_sentence:
            longest_sentence = len(title[sentence].split(' '))

    print("longest sentence: ", longest_sentence)

    print("Counted words: %s" % vocabulary.n_words)
    return article[:max_size], title[:max_size], vocabulary


if __name__ == '__main__':
    article, title, vocabulary = generate_vocabulary()
    print("Article 1337: ")
    print(article[1337])
    print("Title 1337: ")
    print(title[1337])
    print("Word 1337: ")
    print(vocabulary.index2word[1337])
    print("Finished")
