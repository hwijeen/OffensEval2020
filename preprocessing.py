import re
import string
from functools import reduce, partial

import emoji
from wordsegment import load, segment
from transformers import BertTokenizer, RobertaTokenizer, XLMTokenizer, XLNetTokenizer


def compose(*funcs):
    """" Compose functions so that they are applied in chain. """
    return reduce(lambda f, g: lambda x: f(g(x)), funcs[::-1])

def add_capital_signs(text):
    def _has_cap(token):
        return token.lower() != token and token.upper() != token
    def _all_cap(token):
        return token.lower() != token and token.upper() == token
    exceptions = ['@USER', 'URL']
    tokens = text.split()
    tokens = ['<has_cap> ' + t if _has_cap(t) and t not in exceptions else t for t in tokens]
    tokens = ['<all_cap> ' + t if _all_cap(t) and t not in exceptions else t for t in tokens]
    return ' '.join(tokens)

def _limit_pattern(sent, pattern, keep_num):
    if pattern in string.punctuation:
        re_pattern = re.escape(pattern)
    else:
        re_pattern = f'(({pattern})[\s]*)'
        pattern = pattern + ' '
    pattern_regex = re_pattern + '{' + str(keep_num+1) + ',}'
    return re.sub(pattern_regex, lambda match: pattern * keep_num, sent)

def limit_punctuations(sent, keep_num):
    puncs = ['!', '?', '.']
    for p in puncs:
        sent = _limit_pattern(sent, p, keep_num)
    return sent

def limit_mentions(sent, keep_num):
    return _limit_pattern(sent, '@USER', keep_num)

def replace_emojis(sent):
    """ e.g. smiling emoticon -> :smiley_face: """
    return emoji.demojize(sent)

def textify_emojis(sent):
    """ e.g. :smiley_face: -> smiley face"""
    return re.sub(':[\S]+:', lambda match: match.group().replace('_', ' ').replace('-', ' ').replace(':', ''), sent)
    #ret = re.sub(':[\w]+:', lambda match: match.group().replace('_', ' ').replace(':', ''), sent)
    #return '<emoji> ' + ret + ' </emoji>'

def lower_hashtags(sent):
    """ e.g.  #MAGA -> #maga """
    return re.sub('#[\w]+', lambda match: match.group().lower(), sent)

def segment_hashtags(sent):
    """ e.g. #MakeAmericaGreatAgain -> make america great again"""
    return re.sub('#[\w]+', lambda match: ' '.join(segment(match.group())), sent)
    #ret = re.sub('#[\w]+', lambda match: ' '.join(segment(match.group())), sent)
    #return '<hashtag> ' + ret + ' </hashtag>'

def replace_urls(sent):
    return sent.replace('URL', 'http')

def build_preprocess(demojize, textify_emoji, mention_limit, punc_limit, lower_hashtag,
                     segment_hashtag, add_cap_sign):
    if textify_emoji and not demojize:
        raise Exception("textify_emoji is meaningless without demojize")

    funcs = [replace_urls] # default
    if demojize:
        funcs.append(replace_emojis)
    if textify_emoji:
        funcs.append(textify_emojis)
    if mention_limit > 0:
        funcs.append(partial(limit_mentions, keep_num=mention_limit))
    if punc_limit > 0:
        funcs.append(partial(limit_punctuations, keep_num=punc_limit))
    if lower_hashtag:
        funcs.append(lower_hashtags)
    if segment_hashtag:
        load()
        funcs.append(segment_hashtags)
    if add_cap_sign:
        funcs.append(add_capital_signs)
    return compose(*funcs)

# TODO: consider using Config
# TODO: Fix hard code of model names(also in build_model)
def build_tokenizer(model, add_cap_sign, textify_emoji, segment_hashtag, preprocess):
    tokenizer_dict = {'bert': BertTokenizer.from_pretrained('bert-base-uncased')}
                      #'roberta': RobertaTokenizer.from_pretrained('roberta-base'),
                      #'xlm': XLMTokenizer.from_pretrained('xlm-mlm-en-2048'),
                      #'xlnet': XLNetTokenizer.from_pretrained('xlnet-base-cased')}
    if model in tokenizer_dict:
        tokenizer = tokenizer_dict[model]
        tokenizer.add_tokens(['@USER']) # All Transformers models

        if add_cap_sign:
            tokenizer.add_tokens(['<has_cap>', '<all_cap>'])
        if textify_emoji:
            tokenizer.add_tokens(['<emoji>', '</emoji>'])
        if segment_hashtag:
            tokenizer.add_tokens(['<hashtag>', '</hashtag>'])

        if preprocess is not None:
            tokenizer.tokenize = compose(preprocess, tokenizer.tokenize)

    # TODO: when not using bert
    else:
        pass

    return tokenizer
