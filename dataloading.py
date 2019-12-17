import logging
import random
from functools import partial

import torch
from torchtext.data import RawField, Field, TabularDataset, BucketIterator

random.seed(0)
state = random.getstate()

logger = logging.getLogger(__name__)
task_to_col_idx = {'a':2, 'b':3, 'c':4}

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
    def __init__(self, train_path, task, preprocessing, tokenizer,
                 batch_size, device, test_path=None):
        self.task = task
        self.device = device
        self.fields = self.build_field(task, tokenizer, preprocessing)
        train, val = self.build_dataset(train_path)
        self.train_iter, self.val_iter = self.build_iterator((train, val), batch_size, device)
        if test_path is not None:
            test = self.build_dataset(test_path, istest=True)
            self.test_iter = self.build_iterator(test, batch_size, device, istest=True)
        self.build_vocab(train, val)

    def build_field(self, task, tokenizer, preprocessing):
        ID = RawField()
        TWEET = Field(preprocessing=preprocessing, batch_first=True)
        fields = [('id', ID), ('tweet', TWEET), ('NULL', None),
                  ('NULL', None), ('NULL', None)]
        LABEL = Field(sequential=False, unk_token=None, pad_token=None, is_target=True)
        fields[task_to_col_idx[task]] = ('label', LABEL)
        return fields

    def build_dataset(self, data_path, istest=False):
        dataset = TabularDataset(data_path, 'tsv', self.fields,
                                 skip_header=True,
                                 filter_pred=lambda x: x.label != 'NULL')
        if istest:
            return dataset
        else:
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
    """Data format for Transformers model.
    Thie uses TransformersField instead which is an extension of Field"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

    def build_vocab(self, train, val):
        """Does not make vocab for TWEET field as it is given by Transformers models"""
        train.fields['label'].build_vocab(train, val)


def build_data(model, *args, **kwargs):
    if model in {'bert', 'roberta', 'xlm', 'xlnet'}:
        return TransformersData(*args, **kwargs)
    else:
        return Data(*args, **kwargs)
