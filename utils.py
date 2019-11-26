import torch

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

def sequence_mask(lengths, pad=0):
    # make a mask matrix corresponding to given length
    # from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
    row_vector = torch.arange(0, max(lengths), device=lengths.device) # (L,)
    matrix = lengths.unsqueeze(-1) # (B, 1)
    if pad == 1:
        result = row_vector >= matrix # 1 for pad tokens
    else:
        result = row_vector < matrix # 1 for real tokens
    return result # (B, L)

def calc_acc(pred, gold):
    """
    Calculates accuracy between prediction and gold label.
    """
    if isinstance(pred, list) and isinstance(gold, list):
        pred = torch.tensor(pred)
        gold = torch.tensor(gold)
    assert pred.size(0) == gold.size(0)
    N = pred.size(0)
    agree = (pred == gold).sum()
    accuracy = float(agree) / N
    return accuracy

def running_avg(mu, x, alpha):
    return mu + alpha * (x - mu)

def running_avg_list(x_list, x, alpha=0.9):
    try:
        mu = sum(x_list) / len(x_list)
    except ZeroDivisionError:
        mu = 0
    return running_avg(mu, x, alpha)

