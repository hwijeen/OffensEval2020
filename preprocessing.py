from functools import reduce

import emoji

def compose(*functions):
    """"
    Compose functions so that they are applied in chain.
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions[::-1])

def demojize(sent):
    """
    Replace emoticon with predefined :text:.
    """
    return emoji.demojize(sent)

def capitalization(sent):
    pass


