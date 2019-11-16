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
    for name, field in data.train.fields.items():
        if 'label' in name:
            print(f'{name} dictionary: ', field.vocab.stoi)

@lines
def print_shape(batch):
    for name in batch.fields:
        if name == 'id':
            tensor = getattr(batch, name)
            size = len(tensor)
        elif name == 'tweet':
            tensor, lengths = getattr(batch, name)
            size = tensor.size()
        else:
            tensor = getattr(batch, name)
            size = tensor.size()
        print(f'batch.{name} is a {type(tensor)} of size {size}')

def sequence_mask(lengths):
    # make a mask matrix corresponding to given length
    # from https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/python/ops/array_ops.py
    row_vector = torch.arange(0, max(lengths), device=lengths.device) # (L,)
    matrix = lengths.unsqueeze(-1) # (B, 1)
    result = row_vector < matrix # 1 for real tokens
    return result # (B, L)
