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
    9) Add <EOS> token
    :param text: The text to be processed
    :return: The processed text
    """
    text = re.sub("<.*?>", " ", text)  # 1
    text = re.sub('(?<=[^?!.0-9])(?=[.,!?])', ' ', text)  # 4
    text = re.sub('(?=[,])', ' ', text)  # 4
    text = re.sub('(?=\. )', ' ', text)  # 4
    text = text.lower()  # 2
    text = re.sub("[^A-Za-z0-9 .!?,øæå]+", "", text)  # 3
    text = re.sub('[0-9]', '#', text)  # 8
    text = " ".join(text.split())  # 5, 6, 7  - i think
    text = text + " <EOS>"  # 9
    return text


class Article:
    def __init__(self, article_dict, max_words):
        art = article_dict[-1]

        if art is None:
            raise ValueError("Article is of type None")
        if "tags" not in art:
            raise ValueError("Tags not present")
        if "title" not in art:
            raise ValueError("Title not present")
        if "body" not in art:
            raise ValueError("Body not present")

        tags = art["tags"]
        self.tags = []
        for obj in tags:
            if "urlPattern" not in obj:
                continue
            self.tags.append(obj["urlPattern"])  # urlPattern vs. displayName
        self.title = process_text(art["title"])
        self.body = process_text(art["body"])
        if len(self.body.split(" ")) > max_words:
            raise ValueError("body too large")

    def __str__(self):
        text = "Tags: \n" + self.tags.__str__() + "\n"
        text += "Title: \n" + self.title + "\n"
        text += "Body: \n" + self.body + "\n"
        return text

    def __repr__(self):
        self.__str__()


def has_tag(article, tag):
    return tag in article.tags


def get_articles_from_pickle_file(path, max_words=150):
    articles = []
    with open(path, 'rb') as f:
        print("Loading data")
        data = pickle.load(f)
        print("Done loading")
        errors = 0
        non_errors = 0
        for key, value in data.items():
            try:
                articles.append(Article(value, max_words))
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
    tag = "politi"
    max_words = 150

    articles = get_articles_from_pickle_file('../data/articles2_nor/total.pkl', max_words)
    filtered = filter_list_with_single_tag(articles, tag)
    save_articles_for_single_tag(filtered, tag, '../data/articles2_nor/')

    print("Done")
