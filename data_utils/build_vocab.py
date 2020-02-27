"""Builds ngram vocab with berttok or char tok"""
import os
import argparse
from collections import Counter

import emoji
from transformers import BertTokenizer


def examplify(line):
    fields = line.split('\t')
    try:
        id_, tweet, label = fields
        return {'id': id_, 'tweet': tweet, 'label':label}
    except:
        return fields # formatting error

def remove_header(examples):
    header = examples.pop(0)
    header = '\t'.join(list(header.values()))
    return header

def read_examples(filename):
    error_list = []
    examples = []
    lines = open(filename, 'r').readlines()
    for l in lines:
        ex = examplify(l.strip())
        if isinstance(ex, dict):
            examples.append(ex)
        else:
            error_list.append(ex)
    header = remove_header(examples)
    return header, examples, error_list

# NOTE: BertTokenizer replaces emojis with [UNK] token, so char tokenizer removes emojis
def get_tokenizer(token_type):
    if token_type == 'berttok':
        bert_tok = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        return bert_tok.tokenize
    elif token_type =='char':
        emoji_pat = emoji.get_emoji_regexp()
        return lambda tok_list: [char for char in tok_list if emoji_pat.match(char) is None]

def make_ngram(tokens, n):
    ngrams = []
    for i in range(len(tokens) - n):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

def build_ngram_vocab(examples, tokenizer, n):
    counter = Counter()
    for ex in examples:
        tokens = tokenizer(ex['tweet'])
        ngrams = make_ngram(tokens, n)
        counter.update(ngrams)
    return counter

def make_outfname(args):
    dir_ = args.file.split('/')[0]
    fname = f'{args.n}_gram_{args.tokenize}_vocab.txt'
    return os.path.join(dir_, fname)

def write_to_file(ngrams, args):
    outfname = make_outfname(args)
    with open(outfname, 'w') as f:
        for ngram, count in ngrams.most_common():
            print(f'{ngram}\t{count}', file=f)
    return outfname

def change_workdir():
    try:
        os.chdir('data/olid')
    except:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--tokenize', choices=['berttok', 'char'], default='berttok')
    parser.add_argument('--file', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    change_workdir()

    args = parse_args()

    header, examples, error_list = read_examples(args.file)
    tokenizer = get_tokenizer(args.tokenize)
    ngrams = build_ngram_vocab(examples, tokenizer, args.n)
    outfname = write_to_file(ngrams, args)
    print(f'Vocab file written in {outfname}')


