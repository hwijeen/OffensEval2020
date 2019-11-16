import torch

def lines(func):
    """
    Decorator to print lines before and after function execution
    """
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

def sequence_mask(lengths):
    # make a mask matrix corresponding to given length
    # from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
    row_vector = torch.arange(0, max(lengths), device=lengths.device) # (L,)
    matrix = lengths.unsqueeze(-1) # (B, 1)
    # result = row_vector < matrix # 1 for real tokens
    result = row_vector >= matrix # 1 for pad tokens
    return result # (B, L)

def running_avg(mu, x, alpha):
    return mu + alpha * (x - mu)

def running_avg_list(x_list, x, alpha=0.9):
    try:
        mu = sum(x_list) / len(x_list)
    except ZeroDivisionError:
        mu = 0
    return running_avg(mu, x, alpha)

