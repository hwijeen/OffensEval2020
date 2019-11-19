import os
from pandas import read_csv
import pickle
import re
from collections import Counter
from functools import reduce

import emoji
from transformers import BertTokenizer

resources_dir = '../resources/'

def compose(*functions):
    """" Compose functions so that they are applied in chain. """
    return reduce(lambda f, g: lambda x: f(g(x)), functions[::-1])

def demojize(sent):
    """ Replace emoticon with predefined :text:. """
    return emoji.demojize(sent)

#def capitalization(sent):
#    pass

def build_preprocess():
    pass

def build_tokenizer(model, emoji_min_freq=None, hashtag_min_freq=None):        
    if 'bert' in model:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        if emoji_min_freq:
            new_tokens = get_tokens(load_freq_dict('emoji'), emoji_min_freq)
            tokenizer.add_tokens(new_tokens)
        if hashtag_min_freq:
            new_tokens = get_tokens(load_freq_dict('hashtag'), hashtag_min_freq)
            tokenizer.add_tokens(new_tokens)
    return tokenizer

def build_freq_dict(train_corpus, which):
    freq_dict = count(train_corpus, which)
    with open(resources_dir + f'{which}.count', 'wb') as f:
        pickle.dump(freq_dict, f)
    print(f"Built frequency dict {which}.count")

def count(text, which):
    regex = emoji.get_emoji_regexp() if which == 'emoji' else re.compile('#[\w]+')
    tokens = regex.findall(text.lower())
    tokens = list(map(emoji.demojize, tokens)) if which == 'emoji' else tokens
    counts = Counter(tokens)
    return counts

def load_freq_dict(which):
    fpath = os.path.join(resources_dir, f"{which}.count")
    if not os.path.exists(fpath):
        train_corpus = load_corpus()
        build_freq_dict(train_corpus, which)
    with open(fpath, 'rb') as f:
        freq_dict = pickle.load(f)
    return freq_dict

def load_corpus(train_path='../data/olid-training-v1.0.tsv'):
    df = read_csv(train_path, sep='\t', usecols=['tweet'])
    return ' '.join(df['tweet'])

def get_tokens(counter, min_freq):
    return [token for token, freq in counter.items() if freq >= min_freq]

if __name__ == "__main__":
    tokenizer = build_tokenizer('bert', 3, 3)
