from __future__ import unicode_literals, print_function, division

import operator
from io import open


######################################################################
# We'll need a unique index per word to use as the inputs and targets of
# the networks later. To keep track of all this we will use a helper class
# called ``Lang`` which has word → index (``word2index``) and index → word
# (``index2word``) dictionaries, as well as a count of each word
# ``word2count`` to use to later replace rare words.


class Vocabulary:
    def __init__(self):
        self.word2index = {"<SOS>": 0, "<EOS>": 1}
        self.word2count = {"<SOS>": 0, "<EOS>": 0}
        self.index2word = {0: "<SOS>", 1: "<EOS>"}
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


def generate_vocabulary(relative_path, max_size=2000):
    print("Reading lines...")
    article = open(relative_path + '.article.txt', encoding='utf-8').read().strip().split('\n')
    title = open(relative_path + '.title.txt', encoding='utf-8').read().strip().split('\n')
    print("Read %s articles" % len(article))
    print("Read %s title" % len(title))

    vocabulary = Vocabulary()

    longest_sentence = 0

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


class VocabularySizeItem:
    def __init__(self, key, value, num):
        self.key = key
        self.value = value
        self.num = num

    def __str__(self):
        return "%s %s" % (str(self.value), str(self.num))


def count_unk_items(vocab_items):
    vocab_items_single_char = []
    for item in vocab_items:
        if item.num == 1:
            vocab_items_single_char.append(item)

    print(vocab_items_single_char)

    unks = {}
    for item in vocab_items_single_char:
        unk = item.value[0]
        if unk not in unks:
            unks[unk] = []
        unks[unk].append(item)

    total_unks = 0
    for key, value in unks.items():
        print(key, len(value))
        total_unks += len(value)

    print("total = %d" % total_unks)


def get_single_char_list_to_unk(vocab_items):
    vocab_items_single_char = []
    for item in vocab_items:
        if item.num == 1:
            vocab_items_single_char.append(item.value)
    return vocab_items_single_char


def replace_word_with_unk(word):
    return "<UNK" + word[0] + ">"


def save_articles_with_unk(articles, titles, relative_path, to_replace_vocabulary):
    with open(relative_path + '.unk.article.txt', 'w') as f:
        for item in articles:
            words = item.split(" ")
            unked_words = []
            for word in words:
                if word in to_replace_vocabulary:
                    unked_words.append(replace_word_with_unk(word))
                else:
                    unked_words.append(word)
            article = " ".join(unked_words)
            f.write(article)
            f.write("\n")
    with open(relative_path + '.unk.title.txt', 'w') as f:
        for item in titles:
            words = item.split(" ")
            unked_words = []
            for word in words:
                if word in to_replace_vocabulary:
                    unked_words.append(replace_word_with_unk(word))
                else:
                    unked_words.append(word)
            title = " ".join(unked_words)
            f.write(title)
            f.write("\n")


if __name__ == '__main__':
    relative_path_valid = '../data/articles1/valid'
    relative_path_politi = '../data/articles2_nor/politi'
    article, title, vocabulary = generate_vocabulary(relative_path_politi, 5090)
    # print("Article 1337: ")
    # print(article[1337])
    # print("Title 1337: ")
    # print(title[1337])
    # print("Word 1337: ")
    # print(vocabulary.index2word[1337])
    # print("Finished")

    # print(vocabulary.index2word)

    vocab_items = []
    for k, v in vocabulary.index2word.items():
        vocab_items.append(VocabularySizeItem(k, v, vocabulary.word2count[v]))

    # sorted_x = sorted(vocab_items, key=operator.attrgetter('num'), reverse=True)
    # for item in sorted_x:
    #     print(item)

    single_chars_to_unk = get_single_char_list_to_unk(vocab_items)

    print("Unked chars: %d" % len(single_chars_to_unk))

    # save_articles_with_unk(article, title, relative_path_politi, single_chars_to_unk)


# consider one UNK per starting letter (making it 29 different UNK tokens) for every vocabulary[word] = 1
