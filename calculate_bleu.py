import nltk
import re


def read_file(path):
    text = open(path, encoding='utf-8').read()
    titles = []
    output = []
    not_truth = False
    for line in text.split('\n'):
        if line.startswith('='):
            titles.append(line[1:])
            not_truth = True
        elif line.startswith('<') and not_truth:
            output.append(line[1:])
            not_truth = False

    output = clean_text(output)
    titles = tokenize_list(titles)
    output = tokenize_list(output)
    return titles, output


def avg_bleu_score(titles, output):
    avg_bleu = 0
    num_examples = len(titles)
    cc = nltk.translate.bleu_score.SmoothingFunction()

    # Without smoothingfunction we get around 0.3 and with it drops to 0.14. Not sure what is right though
    for i in range(num_examples):
        avg_bleu += nltk.translate.bleu_score.sentence_bleu([titles[i]], output[i], smoothing_function=cc.method4)

    return avg_bleu/num_examples


def clean_text(input_txt):
    output_txt = []
    for line in input_txt:
        line = re.sub(r'\d+', ' ', line)
        line = re.sub(r'[-.]', ' ', line)
        line = re.sub(r'<EOS>', ' ', line)
        output_txt.append(line)
    return output_txt


def tokenize_list(input_list):
    output_list = []
    for line in input_list:
        output_list.append(nltk.wordpunct_tokenize(line))
    return output_list


if __name__ == '__main__':
    path = 'experiments/nhg_test4.txt'
    print("Started extracting titles...")
    titles, output = read_file(path)
    print("Done extracting titles...")
    print("starting to evaluate %d examples..." % len(titles))
    print("Got a BLEU scorer equal: %.4f " % avg_bleu_score(titles, output))
