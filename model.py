import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer

# NUM_CLASS should be specified according to task
NUM_CLASS = 2


class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.out = nn.Linear(self.bert.config.hidden_size, NUM_CLASS)
    
    def forward(self, x):
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
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
    
    def forward(self, x, length):
        """Maps input to last hidden state, to pooler_output, to prediction

        Args:
            x (torch.LongTensor): input of shape (batch_size, seq_length)
        
        Returns:
            x (torch.FloatTensor): logits of shape (batch_size, NUM_CLASS)
        
        """
        x, _ = self.bert(x)         # (batch_size, seq_length, hidden_size)
        x = torch.sum(x, dim=1)     # (batch_size, 1, hidden_size)
        x = x.squeeze()             # (batch_size, hidden_size)
        x = self.out(x)             # (batch_size, NUM_CLASS)
        return x
