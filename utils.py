import os
from datetime import datetime

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# Decorator to print lines before and after function execution
def lines(func):
    def wrapper(*args, **kwargs):
        print("="*80)
        ret = func(*args, **kwargs)
        print("="*80 + '\n')
        return ret
    return wrapper

def sequence_mask(lengths, pad=0, dtype=torch.bool):
    # make a mask matrix corresponding to given length
    # from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
    row_vector = torch.arange(0, max(lengths), device=lengths.device) # (L,)
    matrix = lengths.unsqueeze(-1) # (B, 1)
    if pad == 1:
        result = row_vector >= matrix # 1 for pad tokens
    else:
        result = row_vector < matrix # 1 for real tokens
    if dtype == torch.float:
        result = result.float()
    return result # (B, L)

def calc_acc(pred, gold):
    """
    Calculates accuracy between prediction and gold label.
    """
    return accuracy_score(y_true=gold, y_pred=pred)

def calc_f1(pred, gold, labels=None, pos_label=1, average='macro'):
    """
    Calculates f1 score between prediction and gold label.
    For our task, set average='macro'
    To get score for each category, set labels=[0,1,2] or [0,1] depending on the subtask.
    """
    return f1_score(y_true=gold, y_pred=pred, labels=labels, pos_label=pos_label, average=average)

def calc_prec(pred, gold, labels=None, pos_label=1, average='macro'):
    """
    Calculates precision between prediction and gold label.
    For our task, set average='macro'
    To get score for each category, set labels=[0,1,2] or [0,1] depending on the subtask.
    """
    return precision_score(y_true=gold, y_pred=pred, labels=labels, pos_label=pos_label, average=average)

def calc_rec(pred, gold, labels=None, pos_label=1, average='macro'):
    """
    Calculates recall between prediction and gold label.
    For our task, set average='macro'
    To get score for each category, set labels=[0,1,2] or [0,1] depending on the subtask.
    """
    return recall_score(y_true=gold, y_pred=pred, labels=labels, pos_label=pos_label, average=average)

def conf_matrix(pred, gold, labels=None):
    """
    Computes confusion matrix to evaluate the accuracy of a classification
    Set labels=[0,1,2] or [0,1] depending on the task. ['OFF', 'NOT']
    tn, fp, fn, tp = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
    """
    return confusion_matrix(y_true=gold, y_pred=pred, labels=labels)

def write_pred_to_file(model, data_iter, tokenizer, file_name):
    model.eval()
    data_iter.repeat = False
    ids, tweets, preds, golds, probs = [], [], [], [], []
    with torch.no_grad():
        for batch in data_iter:
            id_ = batch.id
            tweet = [tokenizer.decode(tweet.tolist(), skip_special_tokens=True)\
                      for tweet in batch.tweet[0]]
            pred = model.predict(*batch.tweet).tolist()
            gold = batch.label.tolist()
            prob = model(*batch.tweet).softmax(1).tolist()

            ids += id_
            tweets += tweet
            preds += map(str, pred)
            golds += map(str, gold)
            probs += [' '.join(map(str, p)) for p in prob]
    header = ['id', 'tweet', 'pred', 'gold', 'prob']
    to_write = [header, ids, tweets, preds, golds, probs]
    file_name = write_to_file(file_name, *to_write )
    return file_name

def rename_expname(exp_name):
    """Each of multiple runs with same exp_name will be saved in
    a unique directory, under exp_name."""
    base_exp_name = os.path.join('runs', exp_name)
    now = datetime.now().strftime("%m-%d-%H:%M:%S")
    return os.path.join(base_exp_name, now)

format_to_sep = {'.tsv': '\t', '.csv': ','}
def write_to_file(file_name, header, *args):
    basename, format = os.path.splitext(file_name)
    assert format in format_to_sep
    sep = format_to_sep[format]
    with open(file_name, 'w') as f:
        print(sep.join(header), file=f)
        for line in zip(*args):
            print(sep.join(line), file=f)

def write_summary_to_file(summary, file_name):
    with open(file_name, 'w') as f:
        print('*' * 60, file=f)
        print(summary, file=f)
        print('*' * 60, file=f)

def write_args_to_file(args, file_name='args.bin'):
    torch.save(args, file_name)

def save_model(model, file_name):
    torch.save(model.state_dict(), file_name)

def load_model(model, exp_note, file_name='best_model.pt'):
    exp_name = find_exp(exp_note)
    model_file = os.path.join(exp_name, file_name)
    model.load_state_dict(torch.load(model_file))
    return model

def find_exp(exp_note, run_num=0):
    exp_dir = 'runs/'
    exp_paths = listdir_fullpath(exp_dir)
    exp_notes = ['_'.join(d.split('_')[16:]) for d in exp_paths]
    try:
        exp_id = exp_notes.index(exp_note)
        exp_path = exp_paths[exp_id]
    except ValueError:
        raise Exception(f'No such exp as {exp_note}')
    exp_path = listdir_fullpath(exp_path)[run_num]
    print(f'Accessing exp: {exp_path}')
    return exp_path

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def load_args_from_file(exp_note, file_name='args.bin'):
    exp_path = find_exp(exp_note)
    return torch.load(os.path.join(exp_path, file_name))

def save_tokenizer(tokenizer, dir_name):
    """This writes `added_tokens.json`, `special_tokens_map.json`,
    `vocab.txt`, `tokenizer_config.json` to the directory"""
    tokenizer.save_pretrained(dir_name)
