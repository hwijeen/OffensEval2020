# usage: python merge_all_lang.py --tgt_lang da --data_size 100
import os
import argparse


def make_output_filename(langs):
    dir_ = 'langs'
    filename = '_'.join(langs) + '.tsv'
    return os.path.join(dir_, filename)

def append_lang(line, lang):
    return line.strip() + f'\t{lang}'

def merge_files(files_to_merge):
    all_lines = []
    for lang, filepath in files_to_merge:
        with open(filepath, 'r') as f:
            f.readline() # skip header
            for line in f:
                line = append_lang(line, lang)
                all_lines.append(line)
    return all_lines

def find_files(langs):
    files_to_merge = []
    for lang in langs:
        filenames = os.listdir(lang)
        is_train = lambda f: ('2665' in f) and ('train') in f
        files_to_merge += [(lang, os.path.join(lang, f)) for f in filenames if is_train(f)]
    return files_to_merge

def write_to_file(all_lines, output_filename):
    header = '\t'.join(['id', 'tweet', 'subtask_a', 'lang'])
    with open(output_filename, 'w') as f:
        print(header, file=f)
        for l in all_lines:
            print(l, file=f)

def change_workdir():
    try:
        os.chdir('../data/olid')
    except:
        pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--langs', nargs='+', default=[])
    return parser.parse_args()


if __name__ == "__main__":
    change_workdir()
    args = parse_args()

    files_to_merge = find_files(args.langs)
    print(f'Files to merge:{files_to_merge}')
    all_lines = merge_files(files_to_merge)
    output_filename = make_output_filename(args.langs)
    write_to_file(all_lines, output_filename)
    print(f'Merged file written at {output_filename}')
