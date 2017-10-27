import json


def read_articles(relative_path):
    articles = open(relative_path + '.article.txt', encoding='utf-8').read().strip().split('\n')
    return articles


def count_categories(articles):
    categories = {}
    for article in articles:
        category = article.split(" ")[0]
        if category in categories:
            categories[category] += 1
        else:
            categories[category] = 1
    print(json.dumps(categories, indent=2), flush=True)


if __name__ == '__main__':
    relative_path_ntb = '../data/ntb/ntb_80_6cat'
    articles = read_articles(relative_path_ntb)
    count_categories(articles)
