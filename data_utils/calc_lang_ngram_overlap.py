""" Calculates ngram overlap between two languages, given two ngram vocab files"""
import os
import argparse


def read_ngrams(fpath, k=None):
    ngrams = set()
    lines = [l.strip() for l in open(fpath, 'r')][:k]
    for line in lines:
        ng, cnt = line.split('\t')
        ngrams.add(ng)
    return ngrams

def calc_overlap(src_vocab, tgt_vocab):
    assert len(src_vocab) == len(tgt_vocab), 'vocab size for the two langs must match'
    overlap = len(src_vocab.intersection(tgt_vocab))
    print(src_vocab.intersection(tgt_vocab))
    return overlap / len(src_vocab)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src')
    parser.add_argument('--tgt')
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--append_tgt_lang', action='store_true')
    return parser.parse_args()

def change_workdir():
    try:
        os.chdir('../data/olid')
    except:
        pass

if __name__ == '__main__':
    change_workdir()

    args = parse_args()
    src_vocab = read_ngrams(args.src, args.k)
    tgt_vocab = read_ngrams(args.tgt, args.k)
    overlap_score = calc_overlap(src_vocab, tgt_vocab)
    print(f'Ngram vocab overlap with {args.src} and {args.tgt} is {overlap_score:.2f}')
