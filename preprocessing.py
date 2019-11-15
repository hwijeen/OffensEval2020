from functools import reduce

import emoji

def compose(*functions):
    """"
    Compose functions so that they are applied in chain.
    Note that the order of application is from tail to head.
    """
    return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def demojize(sent):
    """
    Replace emoticon with predefined :text:.
    """
    return emoji.demojize(sent)

def capitalization(sent):
    pass


