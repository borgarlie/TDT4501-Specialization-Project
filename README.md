

# TDT4501-Specialization-Project
Deep Learning for NLP and Robo-Journalism

This repository contains the code used to produce the results seen in [robo-journalism.pdf](robo-journalism.pdf)

## Installing and running the project 

1. Clone this github repo to you machine. 

2. Install pytorch, in this project we use python 3.5.2 and cuda:

```sh
$ pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.0.post4-cp35-cp35m-linux_x86_64.whl 
```
If you are using a different OS, download the wheel file from [pytorch](http://pytorch.org).

3. Install all the required python dependecies using pip3. 

```sh
$ pip3 install -r /path/to/requirements.txt
```

## Execution

Running the baseline model:

```sh
$ python3 experiments/run_experiment.py path/to/config/folder 0
```

```sh
Usage: run_experiment.py
 	[Config folder path]
	[Cuda device]
```

Running the proposed model:
```sh
$ python3 research_experiments/run_research_experiment.py path/to/config/folder 0
```

```sh
Usage: run_research_experiment.py
 	[Config folder path]
	[Cuda device]
```

## Preprocessing and dataset

[preprocess_cnn.py](preprocess/preprocess_cnn.py) and [preprocess_ntb.py](preprocess/preprocess_ntb.py) are two files used to preprocess different datasets. Preprocessing is very different for different datasets, and since the dataset we used for our experiments is not publicly available, we can only point to these files as references on how to proceed.

[preprocess.py](seq2seq_summarization/preprocess.py) can be used to further preprocess a dataset, replacing rare words with UNK tokens (as explained in the [report](robo-journalism.pdf), Section 5.1).

The final preprocessed dataset should consist of two files, one with a article per line and one with the corresponding title per line. The article file should have the suffix `.article.txt`, and the title file should have the suffix `.title.txt`

Example (article): example.article.txt

```txt
this is the first article .
this is the second article .
```

Example (article with categories): example.article.txt

```txt
01000 >>> this is the first article .
00010 >>> this is the second article .
```

Example (title): example.title.txt

```txt
this is the first title
this is the second title
```

## Configuration

The config folder should contain a file named `config.json`, with the following fields (example):

```json
{
  "train" : {
    "dataset" : "../data/dataset_example/dataset",
    "num_articles" : -1,
    "num_evaluate" : 6500,
    "throw" : 1000,
    "with_categories": true,
    "num_epochs" : 20,
    "batch_size" : 32,
    "learning_rate" : 0.001,
    "decay_epoch" : 999,
    "decay_frequency": 5,
    "teacher_forcing_ratio" : 0.9,
    "load" : false,
    "load_file" : "test1.pth.tar"
  },
  "classifier" : {
    "path" : "../classifier/model/classifier1.pth.tar"
  },
  "evaluate" : {
    "expansions" : 3,
    "keep_beams" : 20,
    "return_beams": 5
  },
  "model" : {
    "attention" : true,
    "n_layers" : 1,
    "hidden_size" : 128,
    "dropout_p" : 0.1
  },
  "save" : {
    "save_file" : "test1.pth.tar",
    "best_save_file" : "test1_best.pth.tar",
    "attention_path" : "attention/"
  },
  "log" : {
    "print_every" : 1000,
    "plot_every" : 1000
  },
  "tensorboard" : {
    "log_path" : "../log/example_logfile"
  }
}
```

An example is provided [here](experiments/ntb_test_small/config.json).

Explaination of some of the fields:


`Dataset`: Relative path to the dataset used, without the suffixes (.article.txt and .title.txt)

`num_articles`: Number of articles to include from the dataset, -1 means the whole file.

`num_evaluate`: Number of articles used in the evaluation set

`throw`: Number of articles to not include in either the train or the evaluation set

`with_categories`: true or false, depending on whether or not the dataset includes categories, as shown in the dataset example above.

`classifier/path` Relative path to the saved classifier, as explained below. (Only relevant when running the proposed model)


Other fields in the config are default values, and not all are relevant. Those that are most relevant to play around with are the model parameters, e.g. `hidden_size`, `dropout_p` and `teacher_forcing_ratio`.

## Classifier for the proposed model

To run the proposed model, a classifier needs to be trained and the model needs to be stored. This can be done by running:

```sh
$ python3 classifier/train_classifier.py 0
```

```sh
Usage: train_classifier.py
	[Cuda device]
```

There is no config file for the classifier, but the hyperparameters are set in the main method in [train_classifier.py](classifier/train_classifier.py). The same parameters as used in the stored classifier needs to be set in [run_research_experiment.py](research_experiments/run_research_experiment.py).

When training the classifier, the similar two preprocessed files as shown above for the dataset is expected. The category in front of the articles, e.g. `01000`, should contain the same number of categories (5 in this case) as set in the variable `num_classes`.

## Example results

```txt
>  alle de sju nordmennene kom seg trygt gjennom kvalifiseringen til søndagens <UNKs> i holmenkollen . vanskelige forhold preget fredagens kvalifisering . tett snøvær og varierende vind gjorde oppgaven vrien for kenneth gangnes og resten av verdenseliten . best av dem som måtte kvalifisere seg var polske stefan <UNKh> med et hopp på ###,# meter . han hadde tyske andreas wellinger og østerrikske manuel <UNKp> nærmest seg på resultatlista . tom hilde var best av de norske hopperne .
= samtlige nordmenn klare for kollen renn
< -0.7962527275085449 hoppuken alle norske hopper verdenscup i kollen <EOS>
< -0.8302636827741351 sju nordmenn hopper verdenscup i kollen <EOS>
< -0.8912453651428223 hoppuken alle alle norske hopper i kollen <EOS>
< -0.8985360009329659 hoppuken alle norske hopper i kollen <EOS>
< -0.9145355224609375 alle verste hopper verdenscup i kollen <EOS>
```

```txt
>  en mann i ## årene er sendt til sykehus med alvorlige skader etter en knivstikking i sarpsborg natt til torsdag . ingen er pågrepet i saken . politiet i østfold opplyser på twitter at de søker etter gjerningsmannen på flere adresser i sarpsborg . knivstikkingen skjedde i en leilighet på bede .
= mann alvorlig skadd i knivstikking
< -0.3467409133911133 mann knivstukket i sarpsborg <EOS>
< -0.37392368316650393 mann knivstukket i drammen <EOS>
< -0.5440646580287388 mann knivstukket etter knivstikking i sarpsborg <EOS>
< -0.5894615650177002 mann til sykehus etter knivstikking i sarpsborg <EOS>
< -0.5979729758368598 mann knivstukket til sykehus etter knivstikking i sarpsborg <EOS>
```

The first line (denoted by `>`) is the input article. The second line (denoted by `=`) is the actual headline. The last 5 lines (denoted by `<`) are headlines generated by the model (top 5 beam-search results).

## Notes

This repository contains a lot of code in a lot of different files used to preprocess the data, run experiments and generate the results shown in the report. Here, in this README, we have briefly explained how the preprocessing works and how to run the baseline model and the proposed model. Generating other results, such as BLEU score and visualization of attention weights are out of the scope for this README.

