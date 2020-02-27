""" Calculates ngram overlap between a sample and language, given text and ngram vocab file"""
import os
import re
import argparse

import emoji
from transformers import BertTokenizer


def examplify(line):
    fields = line.split('\t')
    if len(fields) == 3:
        id_, tweet, label = fields
        return {'id': id_, 'tweet': tweet, 'label':label}
    elif len(fields) == 4:
        id_, tweet, label, lang = fields
        return {'id': id_, 'tweet': tweet, 'label':label, 'lang': lang}
    else:
        return fields # formatting error

def textify(example):
    fields = []
    fields.append(example['id'])
    fields.append(example['tweet'])
    fields.append(example['label'])
    if 'lang' in example:
        fields.append(example['lang'])
    if 'overlap_score' in example:
        fields.append(str(example['overlap_score']))
    return '\t'.join(fields)

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

def read_ngrams(fpath, k=None):
    ngrams = set()
    lines = [l.strip() for l in open(fpath, 'r')][:k]
    for line in lines:
        try:
            ng, cnt = line.split('\t')
        except ValueError:
            ng = ' '
            cnt = line.strip()
        ngrams.add(ng)
    return ngrams

# NOTE: BertTokenizer replaces emojis with [UNK] token, so char tokenizer removes emojis
def get_tokenizer(token_type):
    if token_type == 'berttok':
        bert_tok = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        return bert_tok.tokenize
    elif token_type =='char':
        emoji_pat = emoji.get_emoji_regexp()
        return lambda tok_list: [char for char in tok_list if emoji_pat.match(char) is None]

def calc_overlap(sent, ngram_vocab, tokenizer):
    sent_vocab = set(tokenizer(sent))
    overlap = len(sent_vocab.intersection(ngram_vocab))
    k = len(ngram_vocab) # TODO: consider normalization with len(sent_vocab)
    return overlap / k

def append_overlap_score(examples, ngram_vocab, token_type):
    tokenizer = get_tokenizer(token_type)
    for ex in examples:
        score = calc_overlap(ex['tweet'], ngram_vocab, tokenizer)
        ex['overlap_score'] = score
    return examples

def sort_by_overlap_score(examples, size):
    return sorted(examples, key=lambda x: x['overlap_score'], reverse=True)[:size]

def write_to_file(filename, ex_list, header):
    with open(filename, 'w') as f:
        print(header, file=f)
        for ex in ex_list:
            line = textify(ex)
            print(line, file=f)

def make_outfname(args):
     dir_ = 'ngram_samples'
     try:
         src_lang = args.src.split('/')[1].split('-')[1]
     except:
         src_lang = args.src.split('/')[1].split('.')[0]
     if src_lang == 'greek':
         src_lang = 'el'
    if src_lang == 'training':
        src_lang = 'en'
     tgt_lang = args.tgt_ngram.split('/')[0]
     fname = f'{src_lang}_sorted_with_{tgt_lang}_{args.tokenize}_{args.output_size}' + '.tsv'
     return os.path.join(dir_, fname)

def find_train_file(tgt_ngram_path):
    lang = tgt_ngram_path.split('/')[0]
    for f in os.listdir(lang):
        if '2665' in f:
            return os.path.join(lang, f)

def change_workdir():
    try:
        os.chdir('../data/olid')
    except:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', required=True)
    parser.add_argument('--tgt_ngram', required=True)
    parser.add_argument('--ngram_topk', type=int, default=None)
    parser.add_argument('--output_size', type=int, default=2000)
    parser.add_argument('--append_tgt_lang', action='store_false')
    args = parser.parse_args()
    args.tokenize = re.search('berttok|char', args.tgt_ngram).group()
    return args

if __name__ == '__main__':
    change_workdir()
    args = parse_args()

    _, src_examples, _ = read_examples(args.src)
    ngram_vocab = read_ngrams(args.tgt_ngram, args.ngram_topk)
    examples = append_overlap_score(src_examples, ngram_vocab, args.tokenize)
    sorted_examples = sort_by_overlap_score(examples, args.output_size)
    outfname = make_outfname(args)
    header = 'id\ttweet\tlabel\tlang\tngram_overlap'
    if args.append_tgt_lang:
        tgt_file = find_train_file(args.tgt_ngram)
        _, tgt_examples, _ = read_examples(tgt_file)
        sorted_examples += tgt_examples
    write_to_file(outfname, sorted_examples, header)
    print(f'File written at {outfname}')
