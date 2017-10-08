import os
import re


class Article:
    def __init__(self):
        self.text = ""
        self.title = ""

    def clean(self):
        self.text = Article.clean_single(self.text)
        self.title = Article.clean_single(self.title)

    @staticmethod
    def clean_single(txt):
        if len(txt.strip()) < 5:
            raise ValueError
        # TODO: Maybe remove "cnn" from the start of the text if it is there?
        txt = process_text(txt)
        return txt

    def __str__(self):
        return "Title: " + self.title + "\nText: " + self.text

    def __repr__(self):
        return self.__str__()


def process_text(text):
    """
    This function uses multiple steps to process the input properly
    1) Remove all html tags from the text
    2) Lower case every character
    3) Remove all special characters except punctuation, comma, exclamation mark and question mark
    4) Make all these types of punctuations separated from the text (single tokens)
    5) Remove all multi-spaces so that we only have 1 space between each word
    6) Remove possibly space from start and end of text
    7) Replace tab and newlines with space
    8) Replace all numbers with <###>   -- maybe it should be a # per digit instead, as done some other place?
    9) Add space between numbers and words
    :param text: The text to be processed
    :return: The processed text
    """
    text = re.sub('(?<=[^?!.0-9])(?=[.,!?])', ' ', text)  # 4
    text = re.sub(r'(?![0-9])(?<=[.,])(?=[^\s])', r' ', text)  # 4
    text = text.lower()  # 2
    text = re.sub("[^A-Za-z0-9 .!?,øæå]+", " ", text)  # 3
    text = re.sub('[0-9]', '#', text)  # 8
    text = " ".join(text.split())  # 5, 6, 7  - i think
    return text


def read_directory(directory):
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".story"):
            yield os.path.join(directory, file)


def read_content(directory):
    i = 0
    errors = 0
    for filename in read_directory(directory):
        i += 1
        if i % 1000 == 0:
            print(i)
        with open(filename, 'r') as file:
            try:
                yield get_article_from_file(file)
            except ValueError:
                errors += 1
    print("Total articles: %d" % i)
    print("Total errors: %d" % errors)


def get_article_from_file(file):
    first_line_found = False
    highlight_is_next = False
    article = Article()
    for line in file:
        if len(line.strip()) < 2:
            continue
        elif not first_line_found:
            article.text = line
            first_line_found = True
        elif highlight_is_next:
            article.title = line
            break
        elif line.strip().startswith("@highlight"):
            highlight_is_next = True
    article.clean()
    return article


def save_articles(articles, name, relative_path):
    with open(relative_path + name + '.article.txt', 'w') as f:
        for item in articles:
            f.write(item.text)
            f.write("\n")
    with open(relative_path + name + '.title.txt', 'w') as f:
        for item in articles:
            f.write(item.title)
            f.write("\n")


if __name__ == '__main__':
    directory = os.fsencode("../data/cnn2/stories/")
    articles = list(read_content(directory))
    relative_save_path = "../data/preprocessed_cnn/"
    name = "all_test_cnn"
    save_articles(articles, name, relative_save_path)
    print("Number of OK articles: %d" % len(articles))
    print("DONE")
