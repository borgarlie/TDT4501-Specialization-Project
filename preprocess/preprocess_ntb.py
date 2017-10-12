import json
import pickle
import re


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


def is_not_contructive_article(text):
    if text.startswith("tippetips fra ntb"):
        return True
    return False


def clean_ntb(text):
    words = text.split(" ")
    contains_ntb = False
    for k in range(0, 10):
        if "NTB" in words[k].upper() or "NPK" in words[k].upper():
            contains_ntb = True
        if words[k].endswith(":"):
            if contains_ntb:
                words = words[k+1:]
            break
    return " ".join(words)


def clean_copyright(text):
    words = text.split(" ")
    length = len(words)
    for k in range(length-5, length):
        if "\u00a9" in words[k]:
            words = words[0:k]
            break
    if "(NTB)" in words[-1].upper():
        words = words[0:-1]
    return " ".join(words)


def clean_yet_another_ntb(text):
    words = text.split(" ")
    if "ntb" in words[-1] and "." in words[-2]:
        words = words[:-1]
    return " ".join(words)


def shorten_text(text, max_length, min_length):
    words = text.split(" ")
    for i in range(max_length-1, min_length, -1):
        # should break on any of . ? !
        if "." == words[i] or "?" == words[i] or "!" == words[i]:
            words = words[:i+1]
            break
    else:
        raise ValueError("No punctation to stop at when shortening")
    return " ".join(words)


class Article:
    def __init__(self, art, max_words, min_words, min_title):
        if art is None:
            raise ValueError("Article is of type None")
        if "title" not in art:
            raise ValueError("Title not present")
        if "text" not in art:
            raise ValueError("Text not present")

        self.model = "none"
        if "nyhetstype" in art:
            self.model = art["nyhetstype"]
        if self.model != "Nyheter":
            raise ValueError("Nyhetstype should be 'Nyheter'")

        self.title = process_text(art["title"])
        title_length = len(self.title.split(" "))
        if title_length < min_title:
            raise ValueError("Title too small")
        elif title_length > max_words:
            raise ValueError("Title too big")

        text = art["text"]
        text = clean_ntb(text)
        text = clean_copyright(text)
        self.body = process_text(text)
        self.body = clean_yet_another_ntb(self.body)
        article_length = len(self.body.split(" "))
        if article_length > max_words:
            self.body = shorten_text(self.body, max_words, min_words)
            # raise ValueError("body too large")
        elif article_length < min_words:
            raise ValueError("body too small")
        elif article_length <= title_length + 5:
            raise ValueError("Article length is smaller than title length + 5")

        if is_not_contructive_article(self.body):
            raise ValueError("Not constructive article")

    def __str__(self):
        text = "Title: \n" + self.title + "\n"
        text += "Body: \n" + self.body + "\n"
        return text

    def __repr__(self):
        self.__str__()


def count_things_in_article(data):
    things = {}
    for article in data:
        for key, value in article.items():
            if key in things:
                things[key] += 1
            else:
                things[key] = 1
    print(json.dumps(things, indent=2), flush=True)


def count_categories(data):
    categories = {}
    for article in data:
        cat = article['nyhetstype']
        if cat in categories:
            categories[cat] += 1
        else:
            categories[cat] = 1
    print(json.dumps(categories, indent=2), flush=True)


def get_articles_from_pickle_file(path, max_words=100, min_words=25, min_title=4):
    articles = []
    with open(path, 'rb') as f:
        print("Loading data")
        data = pickle.load(f)
        print("Done loading")
        errors = 0
        non_errors = 0
        error_types = {}
        for article in data:
            try:
                articles.append(Article(article, max_words, min_words, min_title))
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
    return articles


def save_articles_for_single_tag(articles, tag, relative_path):
    with open(relative_path + tag + '.article.txt', 'w') as f:
        for item in articles:
            f.write(item.body)
            f.write("\n")
    with open(relative_path + tag + '.title.txt', 'w') as f:
        for item in articles:
            f.write(item.title)
            f.write("\n")


if __name__ == '__main__':
    tag = "ntb_80"
    max_words = 80
    min_words = 25
    min_title = 4
    print("max words: %d" % max_words)
    print("min words: %d" % min_words)
    print("min title: %d" % min_title)

    articles = get_articles_from_pickle_file('../data/ntb/ntb.pkl', max_words, min_words, min_title)
    # filtered = filter_list_with_single_tag(articles, tag)
    save_articles_for_single_tag(articles, tag, '../data/ntb/')
    print("Total articles saved: %d" % len(articles))
    print("Done")
