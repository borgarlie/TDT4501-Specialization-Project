from __future__ import unicode_literals, print_function, division

import operator
from io import open


# Vocabulary for single characters
class Vocabulary:
    def __init__(self):
        self.word2index = {"<": 0, ">": 1}
        self.word2count = {"<": 0, ">": 0}
        self.index2word = {0: "<", 1: ">"}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for character in sentence:
            self.add_character(character)

    def add_character(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def generate_vocabulary(relative_path, max_size=-1):
    print("Reading lines...")
    article = open(relative_path + '.article.txt', encoding='utf-8').read().strip().split('\n')
    title = open(relative_path + '.title.txt', encoding='utf-8').read().strip().split('\n')
    print("Read %s articles" % len(article))
    print("Read %s title" % len(title))

    if max_size == -1:
        max_size = len(article)

    vocabulary = Vocabulary()

    # Problem with this approach: The long term dependency will be A LOT higher since we have to run it for the number
    # of characters in the sentence instead of words. That might also make it slower..
    # even if do not use softmax on a huge vocabulary, because of the fact that backtracking takes many more steps.
    longest_sentence = 0

    print("Counting words...")
    # for sentence in article:
    for sentence in range(0, max_size):
        vocabulary.add_sentence(article[sentence])
        if len(article[sentence]) > longest_sentence:
            longest_sentence = len(article[sentence])
    for sentence in range(0, max_size):
        vocabulary.add_sentence(title[sentence])
        if len(title[sentence]) > longest_sentence:
            longest_sentence = len(title[sentence])

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


def get_list_to_unk(vocab_items, min_freq=3):
    vocab_items_single_char = []
    for item in vocab_items:
        if item.num <= min_freq:
            vocab_items_single_char.append(item.value)
    return vocab_items_single_char


def replace_word_with_unk(word):
    return "<UNK" + word[0] + ">"


def save_articles_with_unk(articles, titles, relative_path, to_replace_vocabulary):
    articles_to_skip = []
    num_unks_10 = 0
    with open(relative_path + '.unk.article.txt', 'w') as f:
        for item in range(0, len(articles)):
            num_unk = 0
            words = articles[item].split(" ")
            unked_words = []
            for word in words:
                if word in to_replace_vocabulary:
                    unked_words.append(replace_word_with_unk(word))
                    num_unk += 1
                else:
                    unked_words.append(word)
            if num_unk >= 10:
                num_unks_10 += 1
                articles_to_skip.append(item)
            else:
                article = " ".join(unked_words)
                f.write(article)
                f.write("\n")
    print("Articles with 10 or more UNK: %d" % num_unks_10)
    with open(relative_path + '.unk.title.txt', 'w') as f:
        for item in range(0, len(titles)):
            if item in articles_to_skip:
                continue
            words = titles[item].split(" ")
            unked_words = []
            for word in words:
                if word in to_replace_vocabulary:
                    unked_words.append(replace_word_with_unk(word))
                else:
                    unked_words.append(word)
            title = " ".join(unked_words)
            f.write(title)
            f.write("\n")


def count_low_length(articles, titles):
    num_less_than_title = 0
    num_abit_more = 0
    num_too_short_article = 0
    num_too_short_title = 0
    for item in range(0, len(articles)):
        if len(articles[item].split(" ")) <= len(titles[item].split(" ")):
            num_less_than_title += 1
        elif len(articles[item].split(" ")) <= len(titles[item].split(" ")) + 10:
            num_abit_more += 1
        elif len(articles[item].split(" ")) < 25:
            num_too_short_article += 1
        elif len(titles[item].split(" ")) < 4:
            num_too_short_title += 1
    print("Articles less than 25 words: %d" % num_too_short_article)
    print("Titles less than 4 words: %d" % num_too_short_title)
    print("Articles with length==title length: %d" % num_less_than_title)
    print("Articles with length less than len(title) + 10: %d" % num_abit_more)


if __name__ == '__main__':
    relative_path_valid = '../data/articles1/valid'
    relative_path_politi = '../data/articles2_nor/politi'
    relative_path_len80 = '../data/articles2_nor/all_len_25to80v3'
    relative_path_len80_skip = '../data/articles2_nor/all_len_25to80_skip_v3'

    article, title, vocabulary = generate_vocabulary(relative_path_len80, -1)

    vocab_items = []
    for k, v in vocabulary.index2word.items():
        vocab_items.append(VocabularySizeItem(k, v, vocabulary.word2count[v]))

    sorted_x = sorted(vocab_items, key=operator.attrgetter('num'), reverse=True)
    for item in sorted_x:
        print(item)

    minimum_frequency = 5
    unked_chars = get_list_to_unk(vocab_items, minimum_frequency)

    print("Unked chars: %d" % len(unked_chars))
    print("Remaining vocab: %d" % (len(vocab_items) - len(unked_chars)))

    count_low_length(article, title)

    # save_articles_with_unk(article, title, relative_path_len80_skip, unked_chars)
