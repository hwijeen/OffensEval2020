# usage: python merge_all_lang.py --tgt_lang da --data_size 100
import os
import argparse


HEADER = '\t'.join(['id', 'tweet', 'subtask_a'])


def make_output_filepath(tgt_lang, tgt_size, src_langs, src_sizes, sort_method):
    dir_ = 'samples'
    filename = f'{tgt_lang}_{tgt_size}'
    for s_lang, s_size in zip(src_langs ,src_sizes):
        filename += f'_{s_lang}_{s_size}'
    filename += f'_{sort_method}.tsv'
    return os.path.join(dir_, filename)

def get_marker(data_size):
    if data_size is None:
        return 'train.tsv'
    else:
        return

def find_tgt_file(tgt_lang, data_size):
    if data_size == 'all':
        marker = 'train.tsv'
    else:
        marker += f'train_{data_size}.tsv'
    for fname in os.listdir(tgt_lang):
        if marker in fname:
            return [os.path.join(tgt_lang, fname)]

def find_src_files(src_lang, src_size, sort_method):
    filepaths = []
    for lang, size in zip(src_lang, src_size):
        for f in os.listdir(lang):
            if (sort_method in f) and (size in f):
                fpath = os.path.join(lang, f)
                filepaths.append(os.path.join(fpath))
    return filepaths

def change_workdir():
    try:
        os.chdir('../data/olid')
    except:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tgt_lang', default='da')
    parser.add_argument('--tgt_size', default='all')
    parser.add_argument('--src_langs', nargs='+', default=['en'])
    parser.add_argument('--src_sizes', nargs='+', default=['all'])
    parser.add_argument('--sort_method', default='l2')
    return parser.parse_args()


if __name__ == "__main__":
    change_workdir()
    args = parse_args()

    src_filepaths = find_src_files(args.src_langs, args.src_sizes,
                                   args.sort_method)
    tgt_filepath = find_tgt_file(args.tgt_lang, args.tgt_size)
    filepaths = src_filepaths + tgt_filepath
    print(f'Source files: {src_filepaths}')

    output_filepath = make_output_filepath(args.tgt_lang, args.tgt_size,
                                           args.src_langs, args.src_sizes,
                                           args.sort_method)

    all_lines = []
    for fpath in filepaths:
        with open(fpath, 'r') as f:
            lines = f.readlines()[1:] # exclude header
            print(f'{fpath} has {len(lines)} lines.')
            all_lines += lines

    with open(output_filepath, 'w') as f:
        print(HEADER, file=f)
        for l in all_lines:
            f.write(l)

    print(f'\nOutput file name: {output_filepath}')
