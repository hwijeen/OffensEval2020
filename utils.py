import os

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

@lines
def print_label_vocab(data):
    print('label dictionary: ', data.train.fields['label'].vocab.stoi)

@lines
def print_shape(batch):
    for name in batch.fields:
        if name == 'NULL':
            continue
        if name == 'id':
            tensor = getattr(batch, name)
            size = len(tensor)
        elif name == 'tweet':
            tensor, lengths = getattr(batch, name)
            size = tensor.size()
        elif name == 'label':
            tensor = getattr(batch, name)
            size = tensor.size()
        print(f'batch.{name} is a {type(tensor)} of size {size}')

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

def calc_prec(pred, gold, labels=None, pos_label=1, average='binary'):
    """
    Calculates precision between prediction and gold label.
    For our task, set average='macro'
    To get score for each category, set labels=[0,1,2] or [0,1] depending on the subtask.
    """
    return precision_score(y_true=gold, y_pred=pred, labels=labels, pos_label=pos_label, average=average)

def calc_recall(pred, gold, labels=None, pos_label=1, average='binary'):
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

def running_avg(mu, x, alpha):
    return mu + alpha * (x - mu)

def running_avg_list(x_list, x, alpha=0.9):
    try:
        mu = sum(x_list) / len(x_list)
    except ZeroDivisionError:
        mu = 0
    return running_avg(mu, x, alpha)

def write_result_to_file(model, data_iter, tokenizer, args_, file_name):
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
    write_to_file(file_name, args_, *to_write )

format_to_sep = {'.tsv': '\t', '.csv': ','}
def write_to_file(file_name, args_, header, *args):
    basename, format = os.path.splitext(file_name)
    assert format in format_to_sep
    sep = format_to_sep[format]
    with open(file_name, 'w') as f:
        print(args_, file=f)
        print(sep.join(header), file=f)
        for line in zip(*args):
            print(sep.join(line), file=f)

def write_summary_to_file(summary, args, file_name):
    with open(file_name, 'w') as f:
        print(args, file=f, end='\n\n')
        print('*' * 60, file=f)
        print(summary, file=f)
        print('*' * 60, file=f)

def write_args_to_file(args, file_name):
    torch.save(args, file_name)

