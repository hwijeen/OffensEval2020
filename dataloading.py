import logging

import random
from functools import partial

import torch
from torchtext.data import RawField, Field, TabularDataset, BucketIterator


logger = logging.getLogger(__name__)


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


class TransformersData:
    """Data format for Transformers model. """
    def __init__(self, preprocessing, tokenizer, batch_size, device,
                 train_path=None, val_path=None, test_path=None):
        self.device = device
        self.fields = self.build_field(tokenizer, preprocessing)
        self.train, self.val, self.test =\
            self.build_dataset(train_path, val_path, test_path)
        self.train_iter, self.val_iter, self.test_iter =\
            self.build_iterator(batch_size, device)
        self.build_vocab()

    def build_field(self, tokenizer, preprocessing):
        """Use custom defined TransformerField which is an extension of torchtext.Field"""
        ID = RawField()
        TWEET = TransformersField(tokenizer, include_lengths=True,
                                  use_vocab=False, batch_first=True,
                                  preprocessing=preprocessing,
                                  tokenize=tokenizer.tokenize,
                                  pad_token=tokenizer.pad_token_id) # id
        LABEL = Field(sequential=False, unk_token=None, pad_token=None)
        fields = [('id', ID), ('tweet', TWEET), ('label', LABEL)]
        return fields

    def build_dataset(self, train_path, val_path, test_path):
        train = val = test = None
        if train_path is not None:
            train = TabularDataset(train_path, 'tsv', self.fields,
                                   skip_header=True)

        if val_path is None:
            random.seed(0)
            state = random.getstate()
            train, val = train.split(split_ratio=0.9, stratified=True,
                                       random_state=state)
        else:
            val = TabularDataset(val_path, 'tsv', self.fields,
                                   skip_header=True)

        if test_path is not None:
            test = TabularDataset(test_path, 'tsv', self.fields,
                                  skip_header=True)
        return train, val, test

    def build_iterator(self, batch_size, device):
        train_iter = val_iter = test_iter = None
        if self.train is not None:
            train_iter = BucketIterator(self.train, batch_size=batch_size,
                                        sort_key=lambda x: len(x.tweet),
                                        sort_within_batch=True, repeat=True,
                                        device=device)
        if self.val is not None:
            val_iter = BucketIterator(self.val, batch_size=batch_size,
                                      sort_key=lambda x: len(x.tweet),
                                      sort_within_batch=True, repeat=True,
                                      device=device, train=False)
        if self.test is not None:
            test_iter = BucketIterator(self.test, batch_size=batch_size,
                                      sort_key=lambda x: len(x.tweet),
                                      sort_within_batch=True, repeat=True,
                                      device=device, train=False)
        return train_iter, val_iter, test_iter

    def build_vocab(self):
        """Does not make vocab for TWEET field as it is given by Transformers models"""
        if self.train is not None and self.val is not None:
            self.train.fields['label'].build_vocab(self.train, self.val)
        elif self.test is not None:
            self.test.fields['label'].build_vocab(self.test)


def build_data(*args, **kwargs):
    return TransformersData(*args, **kwargs)
