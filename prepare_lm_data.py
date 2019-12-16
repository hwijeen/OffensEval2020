import csv
import argparse
from tqdm import tqdm

from preprocessing import build_preprocess

def parse_args():
    parser = argparse.ArgumentParser()

    data = parser.add_argument_group('Data options')
    data.add_argument('--data_path', default='../resources/training.1600000.processed.noemoticon.csv')
    data.add_argument('--out_path', default='../resources/tweet_corpus.txt')


    preprocess = parser.add_argument_group('Preprocessing options')
    preprocess.add_argument('--punctuation')  # not implemented
    preprocess.add_argument('--demojize', action='store_true')
    preprocess.add_argument('--emoji_min_freq', type=int, default=10)
    preprocess.add_argument('--lower_hashtag', action='store_true')
    preprocess.add_argument('--hashtag_min_freq', type=int, default=10)
    preprocess.add_argument('--add_cap_sign', action='store_true')
    preprocess.add_argument('--mention_limit', type=int, default=3)
    preprocess.add_argument('--punc_limit', type=int, default=3)
    preprocess.add_argument('--tokenize', default='bert')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    data_path = args.data_path
    out_path = args.out_path
    demojize = args.demojize
    mention_limit = args.mention_limit
    punc_limit =  args.punc_limit
    lower_hashtag = args.lower_hashtag
    add_cap_sign = args.add_cap_sign
    replace_user = True

    preprocessing = build_preprocess(demojize, mention_limit, punc_limit,
                                     lower_hashtag, add_cap_sign,
                                     replace_user)
    with open(data_path, 'r', errors='ignore') as f_in, open(out_path, 'w') as f_out:
        reader = csv.reader(f_in)
        for line in tqdm(reader):
            tweet = line[-1]
            preprocessed = preprocessing(tweet)
            print(preprocessed, file=f_out)
