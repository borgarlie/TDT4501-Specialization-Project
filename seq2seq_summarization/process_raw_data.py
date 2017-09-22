import pickle
import re
from bs4 import BeautifulSoup


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
    soup = BeautifulSoup(text, "lxml")
    for script in soup(["script", "style"]):  # remove all javascript and stylesheet code
        script.extract()
    text = soup.get_text()
    text = re.sub('(?<=[^?!.0-9])(?=[.,!?])', ' ', text)  # 4
    text = re.sub(r'(?![0-9])(?<=[.,])(?=[^\s])', r' ', text)  # 4
    text = text.lower()  # 2
    text = re.sub("[^A-Za-z0-9 .!?,øæå]+", " ", text)  # 3
    text = re.sub('[0-9]', '#', text)  # 8
    text = " ".join(text.split())  # 5, 6, 7  - i think
    return text


def is_not_contructive_article(text):
    if text.startswith("er du interessert i historie"):
        return True
    return False


class Article:
    def __init__(self, article_dict, max_words, min_words, min_title):
        art = article_dict[-1]

        if art is None:
            raise ValueError("Article is of type None")
        if "title" not in art:
            raise ValueError("Title not present")
        if "body" not in art:
            raise ValueError("Body not present")

        self.model = "none"
        if "model" in art:
            self.model = art["model"]
        if self.model != "story":
            raise ValueError("Model should be 'story'")

        self.preview = True
        if "fields" in art:
            fields = art["fields"]
            if "isPreview" in fields:
                self.preview = fields["isPreview"]
        if self.preview:
            raise ValueError("Preview should be False")

        self.title = process_text(art["title"])
        title_length = len(self.title.split(" "))
        if title_length < min_title:
            raise ValueError("Title too small")

        self.body = process_text(art["body"])
        article_length = len(self.body.split(" "))
        if article_length > max_words:
            raise ValueError("body too large")
        elif article_length < min_words:
            raise ValueError("body too small")
        elif article_length <= title_length + 5:
            raise ValueError("Article length is smaller than title length + 5")

        if is_not_contructive_article(self.body):
            raise ValueError("Not constructive article")

        self.tags = []
        if "tags" in art:
            tags = art["tags"]
            for obj in tags:
                if "urlPattern" not in obj:
                    continue
                self.tags.append(obj["urlPattern"])  # urlPattern vs. displayName

    def __str__(self):
        text = "Tags: \n" + self.tags.__str__() + "\n"
        text += "Title: \n" + self.title + "\n"
        text += "Body: \n" + self.body + "\n"
        return text
        # return "model = %s : preview = %s" % (str(self.model), str(self.preview))

    def __repr__(self):
        self.__str__()


def has_tag(article, tag):
    return tag in article.tags


def get_articles_from_pickle_file(path, max_words=100, min_words=25, min_title=4):
    articles = []
    with open(path, 'rb') as f:
        print("Loading data")
        data = pickle.load(f)
        print("Done loading")
        errors = 0
        non_errors = 0
        for key, value in data.items():
            try:
                articles.append(Article(value, max_words, min_words, min_title))
                non_errors += 1
            except ValueError:
                errors += 1
        print("Done processing data")
        print("total errors = %d" % errors)
        print("Total articles without error = %d" % non_errors)
    return articles


def count_total_tags(articles):
    print("Counting tags: ")
    tags = {}
    for item in articles:
        for tag in item.tags:
            if tag in tags:
                tags[tag] += 1
            else:
                tags[tag] = 1
    print("Done counting tags")
    print("Sorting tags")
    s = [(k, tags[k]) for k in sorted(tags, key=tags.get, reverse=True)]
    for k, v in s:
        print(k, v)
    print("Done printing tags")


def save_articles_for_single_tag(articles, tag, relative_path):
    with open(relative_path + tag + '.article.txt', 'w') as f:
        for item in articles:
            f.write(item.body)
            f.write("\n")
    with open(relative_path + tag + '.title.txt', 'w') as f:
        for item in articles:
            f.write(item.title)
            f.write("\n")


def count_articles_in_max_length_range(articles, start, end):
    range_dict = {}
    for i in range(start, end):
        total = 0
        for item in articles:
            if len(item.body.split(" ")) < i:
                total += 1
        range_dict[i] = total
    for k in sorted(range_dict.keys()):
        print(k, range_dict[k])


def filter_list_with_single_tag(articles, tag):
    tagname_list = []
    for item in articles:
        if has_tag(item, tag):
            tagname_list.append(item)
    return tagname_list


if __name__ == '__main__':
    tag = "all_len_25to80v3"
    max_words = 100
    min_words = 25
    min_title = 4

    articles = get_articles_from_pickle_file('../data/articles2_nor/total.pkl', max_words, min_words, min_title)
    # filtered = filter_list_with_single_tag(articles, tag)
    save_articles_for_single_tag(articles, tag, '../data/articles2_nor/')

    # for item in articles:
    #     print(item)

    print("Done")
