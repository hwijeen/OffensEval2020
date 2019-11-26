import logging

import torch
from torchtext.data import RawField, Field, TabularDataset, BucketIterator

logger = logging.getLogger(__name__)
task_to_col_idx = {'a':2, 'b':3, 'c':4}

# TODO: max_length with torchtext or berttokenizer?
class TransformerField(Field):
    """ Overrides torchtext.data.Field.numericalize to use BertTokenizer.encode
    or RobertaTokenizer.encode"""
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numericalize_func = tokenizer.encode

    def numericalize(self, arr, device=None):
        """To use BertTokenizer.encode"""
        arr, lengths = arr
        lengths = torch.tensor(lengths, dtype=self.dtype, device=device)
        arr = [self.numericalize_func(sent) for sent in arr]
        var = torch.tensor(arr, dtype=self.dtype, device=device)
        var = var.contiguous() # sequential
        return var, lengths


class Data(object):
    """ Holds Datasets, Iterators. """
    def __init__(self, train_path, test_path, task, preprocessing, tokenizer,
                 batch_size, device):
        self.task = task
        self.device = device
        self.fields = self.build_field(task, tokenizer, preprocessing)
        self.train, self.val, self.test = self.build_dataset(train_path,
                                                             test_path)
        self.build_vocab()
        self.train_iter, self.val_iter, self.test_iter = self.build_iterator(
                                                         batch_size, device)

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
        train, val = train_val.split(split_ratio=0.8, stratified=True)
        #test = TabularDataset(test_path, 'tsv', self.fields[:2], skip_header=True) # has no label
        test = TabularDataset(test_path, 'tsv', self.fields, skip_header=True,
                              filter_pred=lambda x: x.label != 'NULL')
        return train, val, test

    def build_vocab(self):
        self.fields['tweet'].build_vocab(self.train, self.val)
        self.train.fields['label'].build_vocab(self.train, self.val)

    # TODO: enable loading only test data
    # TODO: balanced batch needed?
    def build_iterator(self, batch_size, device):
        return BucketIterator.splits((self.train, self.val, self.test),
                                      batch_size=batch_size,
                                      sort_key=lambda x: len(x.tweet),
                                      sort_within_batch=True, repeat=True,
                                      device=device)


class BertData(Data):
    def __init__(self, train_path, test_path, task, preprocessing, tokenizer,
                 batch_size, device):
        self.task = task
        self.device = device
        self.fields = self.build_field(task, tokenizer, preprocessing)
        self.train, self.val, self.test = self.build_dataset(train_path,
                                                             test_path)
        self.build_vocab()
        self.train_iter, self.val_iter, self.test_iter = self.build_iterator(
                                                          batch_size, device)

    def build_field(self, task, tokenizer, preprocessing):
        ID = RawField()
        TWEET = TransformerField(tokenizer, include_lengths=True,
                                 use_vocab=False, batch_first=True,
                                 preprocessing=preprocessing,
                                 tokenize=tokenizer.tokenize,
                                 init_token=tokenizer._cls_token,
                                 eos_token=tokenizer._sep_token,
                                 pad_token=tokenizer.pad_token,
                                 unk_token=tokenizer.unk_token)
        fields = [('id', ID), ('tweet', TWEET), ('NULL', None),
                  ('NULL', None), ('NULL', None)]
        LABEL = Field(sequential=False, unk_token=None, pad_token=None)
        fields[task_to_col_idx[task]] = ('label', LABEL)
        return fields

    def build_vocab(self):
        self.train.fields['label'].build_vocab(self.train, self.val)

def build_data(model, *args, **kwargs):
    if 'bert' in model:
        return BertData(*args, **kwargs)
    else:
        return Data(*args, **kwargs)


# QUESTION: OOV rate?
if __name__ == "__main__":
    from transformers import *
    from utils import *

    train_path = '../data/olid-training-v1.0.tsv'
    test_path = '../data/testset-levela.tsv' # same for a, b, c
    task = 'a'
    batch_size = 32
    cuda = True
    preprocessing = None
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Build data object
    data = build_data('bert', train_path, test_path, task, preprocessing, bert_tok,
                      batch_size, device=cuda)

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

