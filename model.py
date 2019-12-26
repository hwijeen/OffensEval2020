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


class PoolClassifier(CLSClassifier):
    def __init__(self, transformer_model, n_class, time_pooling, layer_pooling, layer):
        super().__init__(transformer_model, n_class)
        pool_time = {'avg': self.avg_pool, 'max': self.max_pool, 'max_avg': self.max_avg_pool}
        pool_layer = {'avg': self.avg_layer, 'max': self.max_layer,
                      'cat': self.concat_layer, 'weight': self.weighted_avg_layer}
        self.layer = layer
        self.time_pooling = pool_time[time_pooling]
        self.layer_pooling = pool_layer[layer_pooling]
        self.hidden_weights = nn.Linear(len(layer), 1)
        self.single_hidden_size = self.model.config.hidden_size * 2 if time_pooling == 'max_avg' \
            else self.model.config.hidden_size
        self.hidden_size = self.single_hidden_size * len(layer) if layer_pooling == 'cat' \
            else self.single_hidden_size
        self.out = nn.Linear(self.hidden_size, n_class)

    def avg_pool(self, x, length):
        x = x.masked_fill(sequence_mask(length, pad=1).unsqueeze(-1), 0.0)
        x = torch.sum(x, dim=1)  # (batch_size, 1, hidden_size)
        x = x.squeeze()  # (batch_size, hidden_size)
        x = x / length.unsqueeze(-1).float()
        return x

    def max_pool(self, x, length):
        x = x.masked_fill(sequence_mask(length, pad=1).unsqueeze(-1), 0.0)
        x = torch.max(x, dim=1).values
        return x

    def max_avg_pool(self, x, length):
        x_mean = self.avg_pool(x, length)
        x_max = self.max_pool(x, length)
        return torch.cat([x_mean, x_max], dim=1)

    def avg_layer(self, hiddens):
        avg_h = torch.stack(hiddens).mean(dim=0)
        return avg_h

    def weighted_avg_layer(self, hiddens):
        avg_h = self.hidden_weights(torch.stack(hiddens, dim=-1))
        return avg_h.squeeze(-1)

    def max_layer(self, hiddens):
        max_h = torch.stack(hiddens).max(dim=0).values
        return max_h

    def concat_layer(self, hiddens):
        return torch.cat(hiddens, dim=1)

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
            _, cls, hidden_states = self.model(x, attention_mask=x_mask)
            # hidden_states : length 13 tuple of tensors (batch_size, max_length, hidden_size)
            if len(self.layer) == 1:
                x = self.time_pooling(hidden_states[self.layer[0]], length)
            else:
                x = self.layer_pooling([self.time_pooling(hidden_states[layer], length)
                                        for layer in self.layer])
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
    else:
        model = PoolClassifier(base_model, n_class, time_pooling, layer_pooling, layer)
    return model.to(device)
