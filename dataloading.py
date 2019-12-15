import logging
from functools import partial
import numpy as np
import torch
from torchtext.data import RawField, Field, TabularDataset, BucketIterator, Batch

logger = logging.getLogger(__name__)
task_to_col_idx = {'a':2, 'b':3, 'c':4}


def load_offensive_list():
    resources_dir = '../resources/'
    with open(resources_dir + 'offensive_words.txt', 'r') as f:
        words = [w.rstrip() for w in f.readlines()]
    return set(words)


class MaskedDataIterator(BucketIterator):
    def __init__(self, dataset, batch_size, **kwargs):
        mask_offensive = kwargs.pop('mask_offensive')
        mask_random = kwargs.pop('mask_random')
        super().__init__(dataset, batch_size, **kwargs)
        self.mask = max(mask_offensive, mask_random) if self.train else 0.0
        self.offensive_list = load_offensive_list() if mask_offensive > 0 else None
        self.masking_ft = partial(self.replace_random, word_set=self.offensive_list, p=self.mask)

    def replace(self, from_, to_, p):
        # replace a word <from_> with probability p to token <to_>
        if np.random.uniform(0, 1) < p:
            return to_
        else:
            return from_

    def replace_random(self, tokens, p, word_set=None, mask='[UNK]'):
        if p == 0:
            return tokens
        if word_set is None:
            masked_tokens = [self.replace(from_=t, to_=mask, p=p) for t in tokens]
        else:
            masked_tokens = [self.replace(from_=t, to_=mask, p=p) if t in word_set else t for t in tokens]
        return masked_tokens

    def __iter__(self):
        while True:
            self.init_epoch()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                for example in minibatch:
                    example.tweet = self.masking_ft(example.tweet)
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                if self.sort_within_batch:
                    # NOTE: `rnn.pack_padded_sequence` requires that a minibatch
                    # be sorted by decreasing order, which requires reversing
                    # relative to typical sort keys
                    if self.sort:
                        minibatch.reverse()
                    else:
                        minibatch.sort(key=self.sort_key, reverse=True)
                yield Batch(minibatch, self.dataset, self.device)
            if not self.repeat:
                return


class TransformersField(Field):
    """ Overrides torchtext.data.Field.process to use Tokenizer.encode as a numericalization function"""
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numericalize_func = partial(tokenizer.encode, add_special_tokens=True)

    def process(self, batch, device=None):
        """Overrides to use numericalize_func first and then pad.
        Note that numericalize_func is used instead of numericalize"""
        ids_list = [self.numericalize_func(ex, device=device) for ex in batch]
        padded, lengths = self.pad(ids_list)
        padded = torch.tensor(padded, device=device)
        lengths = torch.tensor(lengths, device=device)
        if self.sequential:
            padded = padded.contiguous()
        return padded, lengths


class Data(object):
    """ Holds Datasets, Iterators. """
    def __init__(self, train_path, test_path, task, preprocessing, tokenizer,
                 batch_size, device, mask_offensive, mask_random):
        self.task = task
        self.device = device
        self.mask_offensive = mask_offensive
        self.mask_random = mask_random
        self.fields = self.build_field(task, tokenizer, preprocessing)
        self.train, self.val, self.test = self.build_dataset(
            train_path, test_path)
        self.build_vocab()
        self.train_iter, self.val_iter, self.test_iter =\
            self.build_iterator(batch_size, device)

    def build_field(self, task, tokenizer, preprocessing):
        ID = RawField()
        TWEET = Field(preprocessing=preprocessing,
                      batch_first=True)
        fields = [('id', ID), ('tweet', TWEET), ('NULL', None),
                  ('NULL', None), ('NULL', None)]
        LABEL = Field(sequential=False, unk_token=None, pad_token=None, is_target=True)
        fields[task_to_col_idx[task]] = ('label', LABEL)
        return fields

    # TODO: Check stratified split is correct
    def build_dataset(self, train_path, test_path):
        train_val = TabularDataset(train_path, 'tsv', self.fields,
                                   skip_header=True,
                                   filter_pred=lambda x: x.label != 'NULL')
        train, val = train_val.split(split_ratio=0.9, stratified=True)
        test = TabularDataset(test_path, 'tsv', self.fields, skip_header=True,
                              filter_pred=lambda x: x.label != 'NULL')
        return train, val, test

    def build_vocab(self):
        self.fields['tweet'].build_vocab(self.train, self.val)
        self.train.fields['label'].build_vocab(self.train, self.val)

    # TODO: enable loading only test data
    # TODO: balanced batch needed?
    def build_iterator(self, batch_size, device):
        return MaskedDataIterator.splits((self.train, self.val, self.test),
                                         batch_size=batch_size,
                                         sort_key=lambda x: len(x.tweet),
                                         sort_within_batch=True, repeat=True,
                                         device=device, mask_offensive=self.mask_offensive,
                                         mask_random=self.mask_random)


class TransformersData(Data):
    def __init__(self, train_path, test_path, task, preprocessing, tokenizer,
                 batch_size, device, mask_offensive, mask_random):
        self.task = task
        self.device = device
        self.mask_offensive = mask_offensive
        self.mask_random = mask_random
        self.fields = self.build_field(task, tokenizer, preprocessing)
        self.train, self.val, self.test = self.build_dataset(
            train_path, test_path)
        self.build_vocab()
        self.train_iter, self.val_iter, self.test_iter =\
            self.build_iterator(batch_size, device)

    def build_field(self, task, tokenizer, preprocessing):
        ID = RawField()
        TWEET = TransformersField(tokenizer, include_lengths=True,
                                  use_vocab=False, batch_first=True,
                                  preprocessing=preprocessing,
                                  tokenize=tokenizer.tokenize,
                                  pad_token=tokenizer.pad_token_id) # id
        fields = [('id', ID), ('tweet', TWEET), ('NULL', None),
                  ('NULL', None), ('NULL', None)]
        LABEL = Field(sequential=False, unk_token=None, pad_token=None)
        fields[task_to_col_idx[task]] = ('label', LABEL)
        return fields

    def build_vocab(self):
        self.train.fields['label'].build_vocab(self.train, self.val)

def build_data(model, *args, **kwargs):
    if model in {'bert', 'roberta', 'xlm', 'xlnet'}:
        return TransformersData(*args, **kwargs)
    else:
        return Data(*args, **kwargs)
