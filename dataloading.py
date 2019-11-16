import os
import logging

import torch
from torchtext.data import RawField, Field, TabularDataset, BucketIterator

# TODO: logging
logger = logging.getLogger(__name__)

task_to_col_idx = {'A':2, 'B':3, 'C':4}

class BERTField(Field):
    """
    Overrides torchtext.data.Field.numericalize to use BertTokenizer.encode
    """
    def __init__(self, numericalize_func, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numericalize_func = numericalize_func

    def numericalize(self, arr, device=None):
        arr, lengths = arr
        lengths = torch.tensor(lengths, dtype=self.dtype, device=device)
        arr = [self.numericalize_func(sent) for sent in arr]
        var = torch.tensor(arr, dtype=self.dtype, device=device)
        var = var.contiguous() # sequential
        return var, lengths


# TODO: MAXLEN in tweet field
class Data(object):
    """
    Holds Datasets, Iterators.
    """
    def __init__(self, data_dir, task, preprocessing, bert_tok, batch_size,
                 device):
        self.train_path = os.path.join(data_dir, 'olid-training-v1.0.tsv')
        self.test_path = os.path.join(data_dir, 'testset-levela.tsv') # same for a, b, c
        self.device = device
        self.build_dataset(task, preprocessing, bert_tok)
        self.build_iterator(batch_size, device)

    # TODO: Strafied split
    def build_dataset(self, task, preprocessing, bert_tok):

        def build_field():
            ID = RawField()
            TWEET = BERTField(bert_tok.encode, include_lengths=True,
                              use_vocab=False, batch_first=True,
                              preprocessing=preprocessing,
                              tokenize=bert_tok.tokenize,
                              pad_token=bert_tok.pad_token,
                              unk_token=bert_tok.unk_token)
            fields = [('id', ID), ('tweet', TWEET), ('NULL', None),
                      ('NULL', None), ('NULL', None)]
            LABEL = Field(sequential=False, unk_token=None, pad_token=None)
            fields[task_to_col_idx[task]] = ('label', LABEL)
            return fields

        def build_label_vocab():
            self.train.fields['label'].build_vocab(self.train)

        fields = build_field()
        train_val = TabularDataset(self.train_path, 'tsv', fields,
                                   skip_header=True,
                                   filter_pred=lambda x: x.label is not 'NULL')
        self.train, self.val = train_val.split(split_ratio=0.8)
        build_label_vocab()
        self.test = TabularDataset(self.test_path, 'tsv', fields[:2],
                                   skip_header=True) # has no label

    # TODO: enable loading only test data
    # QUESTION: balanced batch?
    def build_iterator(self, batch_size, device):
        self.train_iter, self.valid_iter, self.test_iter = \
        BucketIterator.splits((self.train, self.val, self.test),
                              batch_size=batch_size,
                              sort_key=lambda x: len(x.tweet),
                              sort_within_batch=True, repeat=True,
                              device=device)


# QUESTION: OOV rate?
if __name__ == "__main__":
    from transformers import *
    from utils import *

    data_dir = 'data/'
    task = 'A'
    batch_size = 32
    device = torch.device('cuda')
    preprocessing = None
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Build data object
    data = Data(data_dir, task, preprocessing, bert_tok, batch_size, device)

    # See how targets are mapped to index
    print_label_vocab(data)

    # Generate batches using data_iter
    for batch in data.train_iter:
        # A batch has multiple attiributes
        id = batch.id # list
        tweet, lengths = batch.tweet # two torch.Tensor
        label = batch.label # torch.Tensor
        # See size of each tensors
        print_shape(batch)

        # Example Usage
        # logits = model(tweet)
        # loss = loss_fn(logits, label)
        # loss.backward()

        # Use tokenizer.convert_ids_to_tokens / decoder to convert word_ids to list / string.
        for idx in range(len(batch)):
            sent = tweet[idx].cpu().numpy().tolist()
            tokens = bert_tok.convert_ids_to_tokens(sent)
            text = bert_tok.decode(sent)
            print(f'\nTweet num: {id[idx]}\nText: {text}\nTokens: {tokens}\nLabel: {label[idx]}')
        break

