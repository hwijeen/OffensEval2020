from functools import reduce

import emoji
from transformers import BertTokenizer

def compose(*funcs):
    """" Compose functions so that they are applied in chain. """
    return reduce(lambda f, g: lambda x: f(g(x)), funcs[::-1])

def demojize(sent):
    """ Replace emoticon with predefined :text:. """
    return emoji.demojize(sent)

#def capitalization(sent):
#    pass

def build_preprocess(no_demojize):
    funcs = []
    if not no_demojize:
        funcs.append(demojize)
    return compose(*funcs)

def build_tokenizer(model):
    if 'bert' in model:
        return BertTokenizer.from_pretrained('bert-base-uncased')

