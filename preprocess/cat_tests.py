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


def count_categories_total(articles):
    cats = [0, 0, 0, 0, 0]
    for article in articles:
        categories = article.split(" ")[0]
        for i in range(0, len(categories)):
            cats[i] += int(categories[i])
    print(cats)


if __name__ == '__main__':
    relative_path_ntb = '../data/ntb/ntb_80_6cat.unk'
    articles = read_articles(relative_path_ntb)
    count_categories(articles)
    count_categories_total(articles)
