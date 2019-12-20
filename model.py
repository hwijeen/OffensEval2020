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
    def __init__(self, transformer_model, n_class, time_pooling, layer_pooling, layer):
        super().__init__(transformer_model, n_class)
        self.layer = layer
        self.time_pooling = time_pooling
        self.layer_pooling = layer_pooling
        self.hidden_size = self.model.config.hidden_size * len(layer)
        self.out = nn.Linear(self.hidden_size, n_class)

    def avg_pool(self, x, length):
        x = x.masked_fill(sequence_mask(length, pad=1).unsqueeze(-1), 0.0)
        x = torch.sum(x, dim=1)  # (batch_size, 1, hidden_size)
        x = x.squeeze()  # (batch_size, hidden_size)
        x = x / length.unsqueeze(-1).float()
        return x

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
            _, _, hidden_states = self.model(x, attention_mask=x_mask)
            x = torch.cat([self.avg_pool(hidden_states[layer], length)
                           for layer in self.layer], dim=1)
        except: # xlm, xlnet
            x = self.model(x, attention_mask=x_mask)        # (batch_size, seq_length, hidden_size)
            x = x[0]
        x = self.out(x)                                     # (batch_size, NUM_CLASS)
        return x


# TODO: fix hardcoding of model names(need to be compatible with preprocessing)
def build_model(task, model, time_pooling, layer_pooling, layer,
                new_num_tokens, device, **kwargs):
    n_class = task_to_n_class[task]
    if model == 'bert':
        base_model = BertModel.from_pretrained('bert-base-uncased',
                                               output_hidden_states=True,
                                               **kwargs)
        base_model.resize_token_embeddings(new_num_tokens)
    elif model == 'roberta':
        base_model = RobertaModel.from_pretrained('roberta-base',
                                                  output_hidden_states=True,
                                                  **kwargs)
        base_model.resize_token_embeddings(new_num_tokens)
    elif model =='xlm':
        base_model = XLMModel.from_pretrained('xlm-mlm-en-2048')
        base_model.resize_token_embeddings(new_num_tokens)
    elif model == 'xlnet':
        base_model = XLNetModel.from_pretrained('xlnet-base-cased')
        base_model.resize_token_embeddings(new_num_tokens)
    else:
        pass

    if time_pooling == 'cls':
        model = CLSClassifier(base_model, n_class)
    elif time_pooling == 'avg':
        model = AvgPoolClassifier(base_model, n_class, time_pooling, layer_pooling, layer)
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