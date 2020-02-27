# usage: python train_test_split.py da/offenseval-da-training-v1.tsv
import os
import sys
import random
import pdb


TEST_RATIO = 0.1


def get_fnames(filename, num_train):
    dirname, filename = os.path.split(filename)
    fname, ext = os.path.splitext(filename)
    if num_train is None:
        train_fname = os.path.join(dirname, fname + '-train' + ext)
        test_fname = os.path.join(dirname, fname + '-test' + ext)
    else:
        train_fname = os.path.join(dirname, fname + f'-train_{num_train}' + ext)
        test_fname = os.path.join(dirname, fname + '-test' + ext)

    return train_fname, test_fname

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

def divide_examples(examples):
    off_list = []
    not_list = []
    for ex in examples:
        if ex['label'] == 'OFF':
            off_list.append(ex)
        else:
            not_list.append(ex)
    return off_list, not_list

def split_examples(ex_list, test_ratio, num_ex):
    test_size = int(len(ex_list) * test_ratio)
    random.shuffle(ex_list) # in place
    test_ex = ex_list[:test_size]
    train_ex = ex_list[test_size:]
    if num_ex is not None:
        train_ex = train_ex[:num_ex]
    return train_ex, test_ex

def stratified_ratio(num_train, off_portion):
    if num_train is None:
        return None, None
    off_num = int(num_train * off_portion/100)
    not_num = num_train - off_num
    return off_num, not_num

def examplify(line):
    fields = line.split('\t')
    try:
        id_, tweet, label = fields
        return {'id': id_, 'tweet': tweet, 'label':label}
    except:
        return fields # formatting error

def textify(example):
    id_ = example['id']
    tweet = example['tweet']
    label = example['label']
    return '\t'.join([id_, tweet, label])

def write_to_file(filename, ex_list, header):
    with open(filename, 'w') as f:
        print(header, file=f)
        for ex in ex_list:
            line = textify(ex)
            print(line, file=f)

if __name__ == '__main__':

    filename = sys.argv[1]
    try:
        num_train = int(sys.argv[2])
    except:
        num_train = None # no downsample
    print(f'Input file: {filename}')


    header, examples, error_list = read_examples(filename)
    off_list, not_list = divide_examples(examples)
    off_portion = len(off_list) / len(examples) * 100
    not_portion = len(not_list) / len(examples) * 100
    print(f'\tOFF portion: {off_portion:.2f}%')
    print(f'\tNOT portion: {not_portion:.2f}%')

    off_num, not_num = stratified_ratio(num_train, off_portion)
    train_off, test_off = split_examples(off_list, TEST_RATIO, off_num)
    train_not, test_not = split_examples(not_list, TEST_RATIO, not_num)
    print(f'Original file size: {len(examples)}')


    train = train_off + train_not
    test = test_off + test_not
    train_portion = len(train) / len(examples) * 100
    test_portion = len(test) / len(examples) * 100
    print(f'Train size: {len(train)}({train_portion:.2f}%)')
    print(f'\tOFF: {len(train_off)}')
    print(f'\tNOT: {len(train_not)}')
    print(f'Test size: {len(test)}({test_portion:.2f}%)')
    print(f'\tOFF: {len(test_off)}')
    print(f'\tNOT: {len(test_not)}', end='\n\n')

    train_fname, test_fname = get_fnames(filename, num_train)
    write_to_file(train_fname, train, header=header)
    write_to_file(test_fname, test, header=header)

    print('='*80)
    print('Num SKipped lines: ', len(error_list))
    #print('Skipped lines: ', error_list)
    print(f'Output filename: {train_fname}')
    print(f'Output filename: {test_fname}', end='\n\n')
