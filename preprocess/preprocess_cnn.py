import json
import os
import re


class Article:
    def __init__(self):
        self.text = ""
        self.title = ""

    def clean(self):
        self.text = Article.clean_single(self.text, body=True)
        self.title = Article.clean_single(self.title, body=False)

    @staticmethod
    def clean_single(txt, body=True):
        txt = process_text(txt)
        words = txt.split(" ")
        if len(words) > max_words:
            for i in range(max_words-1, min_words, -1):
                if "." in words[i] or "!" in words[i] or "?" in words[i]:
                    words = words[:i+1]
                    break
            else:
                raise ValueError("No stop token to stop at when shortening")
        txt = ' '.join(words)
        if txt.count(' ') < min_words and body:
            raise ValueError("body too small")
        # print(txt.count(' '))
        return txt

    def __str__(self):
        text = "Title: \n" + self.title + "\n"
        text += "Text: \n" + self.text + "\n"
        return text

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
    no_split_list = ['U.S', 'U.N', 'U.K', 'L.A', 'J.K']  # Seems like they have to be the same length
    no_split_string = str(no_split_list).strip('[]').replace(',', '|').replace('\'', '').replace(' ', '')

    text = re.sub(".*--", "", text, count=1)  # Removing cnn from start of text
    if text.startswith('(CNN)'):
        text = re.sub('\(CNN\)', '', text, count=1)
    text = re.sub(r'(?<=[^?!.0-9{}])(?=[.,!?])'.format(no_split_string), ' ', text)  # 4
    text = re.sub(r'(?![0-9])(?<=[.,])(?=[^\s])', r' ', text)  # 4
    text = re.sub(r'(?<={})(?=[^\s])'.format(no_split_string), r' ', text)  # space after no split list
    text = text.lower()  # 2
    text = re.sub('[^A-Za-z0-9 .!?,øæå]+', '', text)  # 3
    text = re.sub(r'((?<=[a-z])(?=[.]))|((?=[a-z])(?<=[.]))(?=[^\s])', r' ', text)  # space a-z.a-z
    text = re.sub(r'((?=[0-9])(?<=[a-z]))|((?=[a-z])(?<=[0-9]))(?=[^\s])', r' ', text)  # space 0-9a-z
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
    non_errors = 0
    error_types = {}

    for filename in read_directory(directory):
        i += 1
        # if i % 1000 == 0:
        #     print(i)
        with open(filename, 'r') as file:
            try:
                yield get_article_from_file(file)
                non_errors += 1
            except ValueError as err:
                err = err.__str__()
                errors += 1
                if err in error_types:
                    error_types[err] += 1
                else:
                    error_types[err] = 1
    print("Done processing data")
    print("total errors = %d" % errors)
    print("Total articles without error = %d" % non_errors)
    print("Error types: ")
    print(json.dumps(error_types, indent=2), flush=True)
    # print("Total articles: %d" % i)
    # print("Total errors: %d" % errors)


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
        elif article.text.count(' ') < max_words:
            article.text += line
    article.clean()

    if article.title.count(' ') < min_title:
        raise ValueError("Title too small")

    if article.text.count(' ') <= article.title.count(' ') + 10:
        raise ValueError("Article length is smaller than title length + 10")
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
    max_words = 80
    min_words = 25
    min_title = 4
    print("max words: %d" % max_words)
    print("min words: %d" % min_words)
    print("min title: %d" % min_title)

    cnn_directory = os.fsencode("../data/cnn2/stories/")
    dailymail_directory = os.fsencode("../data/dailymail/stories/")
    articles1 = list(read_content(cnn_directory))
    articles2 = list(read_content(dailymail_directory))
    articles = articles1 + articles2

    relative_save_path = "../data/preprocessed_combined/"
    name = "all_test"
    save_articles(articles, name, relative_save_path)
    print("Number of OK articles: %d" % len(articles))
    print("DONE")
