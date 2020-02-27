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


class Data(object):
    """Build field, dataset, and then data iterators. """
    def __init__(self, preprocessing, tokenizer,
                 batch_size, device, train_path=None, test_path=None):
        self.device = device
        self.fields = self.build_field(tokenizer, preprocessing)
        if train_path is not None:
            train, val = self.build_dataset(train_path)
            self.train_iter, self.val_iter = self.build_iterator((train, val), batch_size, device)
            self.build_vocab(train, val)
        if test_path is not None:
            test = self.build_dataset(test_path, istest=True)
            self.test_iter = self.build_iterator(test, batch_size, device, istest=True)
            test.fields['label'].build_vocab(test)

    def build_field(self, tokenizer, preprocessing):
        ID = RawField()
        TWEET = Field(preprocessing=preprocessing, batch_first=True)
        LABEL = Field(sequential=False, unk_token=None, pad_token=None, is_target=True)
        fields = [('id', ID), ('tweet', TWEET), ('label', LABEL)]
        return fields

    def build_dataset(self, data_path, istest=False):
        dataset = TabularDataset(data_path, 'tsv', self.fields, skip_header=True)
        if istest:
            return dataset
        else:
            random.seed(0)
            state = random.getstate()
            return dataset.split(split_ratio=0.9, stratified=True, random_state=state)

    def build_vocab(self, train, val):
        train.fields['tweet'].build_vocab(train, val)
        train.fields['label'].build_vocab(train, val)

    # TODO: enable loading only test data
    def build_iterator(self, dataset, batch_size, device, istest=False):
        if istest:
            return BucketIterator(dataset, batch_size=batch_size,
                                  sort_key=lambda x: len(x.tweet),
                                  sort_within_batch=True, repeat=True,
                                  device=device)
        else:
            train, val = dataset
            return BucketIterator.splits((train, val), batch_size=batch_size,
                                         sort_key=lambda x: len(x.tweet),
                                         sort_within_batch=True, repeat=True,
                                         device=device)


class TransformersData(Data):
    """Data format for Transformers model. """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def build_vocab(self, train, val):
        """Does not make vocab for TWEET field as it is given by Transformers models"""
        train.fields['label'].build_vocab(train, val)


def build_data(model, *args, **kwargs):
    if model in {'mbert', 'xlm'}:
        return TransformersData(*args, **kwargs)
    else:
        return Data(*args, **kwargs)
