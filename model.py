import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from utils import sequence_mask

# TODO: NUM_CLASS should be specified according to task
# NUM_CLASS = 2
task_to_n_class = {'a':2, 'b':2, 'c':3}

class BertClassifier(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(self.bert.config.hidden_size, n_class)
    
    def forward(self, x, length):
        """Maps input to pooler_output, to prediction

        Args:
            x (torch.LongTensor): input of shape (batch_size, seq_length)
        
        Returns:
            x (torch.FloatTensor): logits of shape (batch_size, NUM_CLASS)
        
        """
        _, x = self.bert(x)     # (batch_size, hidden_size)
        x = self.out(x)         # (batch_size, NUM_CLASS)
        return x


class BertAvgPooling(BertClassifier):
    def __init__(self, n_class):
        super().__init__(n_class)
        # self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
    
    def forward(self, x, length):
        """Maps input to last hidden state, to pooler_output, to prediction

        Args:
            x (torch.LongTensor): input of shape (batch_size, seq_length)
            length (torch.LongTensor): input of shape (batch_size, )
        
        Returns:
            x (torch.FloatTensor): logits of shape (batch_size, NUM_CLASS)
        
        """
        x, _ = self.bert(x)                     # (batch_size, seq_length, hidden_size)
        x = x.masked_fill_(sequence_mask(length).unsqueeze(-1), 0.0)
        x = torch.sum(x, dim=1)                 # (batch_size, 1, hidden_size)
        x = x.squeeze()                         # (batch_size, hidden_size)
        x = x / length.unsqueeze(-1).float()
        # x = self.linear(x)                      # (batch_size, hidden_size)
        x = self.out(x)                         # (batch_size, NUM_CLASS)
        return x

def build_model(task, model, device):
    assert model in ['bert', 'bert_avg']
    n_class = task_to_n_class[task]
    if model == 'bert':
        return BertClassifier(n_class).to(device)
    elif model == 'bert_avg':
        return BertAvgPooling(n_class).to(device)


if __name__ == "__main__":
    from dataloading import Data

    data_dir = '../data/olid/'
    # task = 'A'
    # task = 'B'
    task = 'C'
    batch_size = 32
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    preprocessing = None
    bert_tok = BertTokenizer.from_pretrained('bert-base-uncased')
    model = build_model(task, 'bert')
    # model = build_model(task, 'bert_avg')

    # Build data object
    data = Data(data_dir, task, preprocessing, bert_tok, batch_size, device)

    for batch in data.train_iter:
        id_ = batch.id
        tweet, lengths = batch.tweet
        label = batch.label

        logit = model(tweet, lengths)
        print(logit)
        print(logit.shape)
        break