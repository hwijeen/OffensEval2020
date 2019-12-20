import torch

import torch.nn as nn
from transformers import BertModel, RobertaModel, XLMModel, XLNetModel
from utils import sequence_mask

task_to_n_class = {'a':2, 'b':2, 'c':3}


class CLSClassifier(nn.Module):
    def __init__(self, transformer_model, n_class):
        super().__init__()
        self.model = transformer_model
        self.out = nn.Linear(self.model.config.hidden_size, n_class)
    
    def forward(self, x, length):
        """Maps input to pooler_output, to prediction

        Args:
            x (torch.LongTensor): input of shape (batch_size, seq_length)
        
        Returns:
            x (torch.FloatTensor): logits of shape (batch_size, NUM_CLASS)
        
        """
        x_mask = sequence_mask(length, pad=0, dtype=torch.float)  # (batch_size, max_length)
        _, x = self.model(x, attention_mask=x_mask)  # (batch_size, hidden_size)
        x = self.out(x)                             # (batch_size, NUM_CLASS)
        return x

    def predict(self, x, length):
        logits = self(x, length)
        return logits.argmax(1)


class AvgPoolClassifier(CLSClassifier):
    def __init__(self, transformer_model, n_class):
        super().__init__(transformer_model, n_class)

    def forward(self, x, length):
        """Maps input to last hidden state, to pooler_output, to prediction

        Args:
            x (torch.LongTensor): input of shape (batch_size, seq_length)
            length (torch.LongTensor): input of shape (batch_size, )
        
        Returns:
            x (torch.FloatTensor): logits of shape (batch_size, NUM_CLASS)
        
        """
        x_mask = sequence_mask(length, pad=0, dtype=torch.float)  # (batch_size, max_length)
        # TODO: clean this hack
        try: # bert, roberta
            x, _ = self.model(x, attention_mask=x_mask)                     # (batch_size, seq_length, hidden_size)
        except: # xlm, xlnet
            x = self.model(x, attention_mask=x_mask)                     # (batch_size, seq_length, hidden_size)
            x = x[0]
        x = x.masked_fill_(sequence_mask(length, pad=1).unsqueeze(-1), 0.0)
        x = torch.sum(x, dim=1)                 # (batch_size, 1, hidden_size)
        x = x.squeeze()                         # (batch_size, hidden_size)
        x = x / length.unsqueeze(-1).float()
        # x = self.linear(x)                      # (batch_size, hidden_size)
        x = self.out(x)                         # (batch_size, NUM_CLASS)
        return x


# TODO: fix hardcoding of model names(need to be compatiable with preprocessing)
def build_model(task, model, pooling, new_num_tokens, device, **kwargs):
    n_class = task_to_n_class[task]
    if 'checkpoint' in model:
        base_model = BertModel.from_pretrained(model, **kwargs)
        base_model.resize_token_embeddings(new_num_tokens)

    if model == 'bert':
        base_model = BertModel.from_pretrained('bert-base-uncased', **kwargs)
        base_model.resize_token_embeddings(new_num_tokens)
    elif model == 'roberta':
        base_model = RobertaModel.from_pretrained('roberta-base', **kwargs)
        base_model.resize_token_embeddings(new_num_tokens)
    elif model =='xlm':
        base_model = XLMModel.from_pretrained('xlm-mlm-en-2048')
        base_model.resize_token_embeddings(new_num_tokens)
    elif model == 'xlnet':
        base_model = XLNetModel.from_pretrained('xlnet-base-cased')
        base_model.resize_token_embeddings(new_num_tokens)
    else:
        pass

    if pooling == 'cls':
        model = CLSClassifier(base_model, n_class)
    elif pooling == 'avg':
        model = AvgPoolClassifier(base_model, n_class)
    else:
        pass
    return model.to(device)


if __name__ == "__main__":
    from dataloading import Data
    from transformers import BertTokenizer

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