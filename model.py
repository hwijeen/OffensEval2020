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
        self.time_pooling = time_pooling
        self.layer_pooling = layer_pooling
        self.hidden_weights = nn.Linear(len(layer), 1)
        self.single_hidden_size = self.model.config.hidden_size * 2 if time_pooling == 'max_avg' \
            else self.model.config.hidden_size
        self.hidden_size = self.single_hidden_size * len(layer) if layer_pooling == 'cat' \
            else self.single_hidden_size
        self.out = nn.Linear(self.hidden_size, n_class)

    def time_pool(self, x, length):
        pool_time = {'avg': self.avg_pool, 'max': self.max_pool, 'max_avg': self.max_avg_pool}
        pooling_fn = pool_time[self.time_pooling]
        return pooling_fn(self, x, length)

    def layer_pool(self, hiddens):
        pool_layer = {'avg': self.avg_layer, 'max': self.max_layer,
                      'cat': self.concat_layer, 'weight': self.weighted_avg_layer}
        pooling_fn = pool_layer[self.layer_pooling]
        return pooling_fn(hiddens)

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
                x = self.time_pool(hidden_states[self.layer[0]], length)
            else:
                x = self.layer_pool([self.time_pool(hidden_states[layer], length)
                                     for layer in self.layer])
        except: # xlm, xlnet
            x = self.model(x, attention_mask=x_mask)        # (batch_size, seq_length, hidden_size)
            x = x[0]
        x = self.out(x)                                     # (batch_size, NUM_CLASS)
        return x


class BertCNN(CLSClassifier):
    def __init__(self, transformer_model, n_class, channels, window_size, activation, pooling):
        super().__init__(transformer_model, n_class)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=channels,
                                    kernel_size=(width, self.model.config.hidden_size), stride=1)
                                    for width in window_size])
        activation_map = {'relu': nn.ReLU(), 'lrelu': nn.LeakyReLU(), 'glu': nn.GLU()}
        self.activation = activation_map[activation]
        self.pooling = pooling
        self.hidden_size = 12 * channels * len(window_size)
        if pooling == 'max_avg':
            self.hidden_size *= 2
        self.out = nn.Linear(self.hidden_size, n_class)

    def cnn(self, x, length):
        x = x.masked_fill(sequence_mask(length, pad=1).unsqueeze(-1), 0.0)  # (batch_size, seq_len, hidden_dim)
        conv_outputs = []  # gather convolution outputs here
        for conv in self.convs:
            x = conv(x.unsqueeze(1))       # (batch_size, C, T', 1)
            x = x.squeeze(-1)              # (batch_size, C, T')
            conv_outputs += x
        outputs = torch.cat(conv_outputs, dim=1)
        return outputs

    def max_pool(self, x):
        x = torch.max(x, dim=-1).values   # (batch_size, C)
        return x

    def avg_pool(self, x, length):
        x = torch.sum(x, dim=-1)  # (batch_size, C, 1)
        x = x.squeeze()           # (batch_size, C)
        x = x / length.unsqueeze(-1).float()
        return x

    def single_layer_cnn(self, x, length):
        # x : tensor of size                  (batch_size, C, T', 1)
        x = self.cnn(x, length)             # (batch_size, C, T')
        x = self.activation(x)              # (batch_size, C, T')
        max_x = self.max_pool(x)            # (batch_size, C)
        avg_x = self.avg_pool(x, length)    # (batch_size, C)
        if self.pooling == 'avg':
            x = avg_x
        elif self.pooling == 'max_avg':
            x = torch.cat([avg_x, max_x], dim=1)
        else:
            x = max_x
        return x

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
        try:  # bert, roberta
            _, _, hidden_states = self.model(x, attention_mask=x_mask)
            # hidden_states : length 13 tuple of tensors (batch_size, max_length, hidden_size)
            x = self.concat_layer([self.single_layer_cnn(layer, length)
                                   for layer in hidden_states[1:]])
        except:  # xlm, xlnet
            x = self.model(x, attention_mask=x_mask)  # (batch_size, seq_length, hidden_size)
            x = x[0]
        x = self.out(x)  # (batch_size, NUM_CLASS)
        return x

# TODO: fix hardcoding of model names(need to be compatible with preprocessing)
def build_model(task, model, time_pooling, layer_pooling, layer,
                channels, window_size, activation, cnn_pooling,
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
    elif time_pooling == 'cnn':
        model = BertCNN(base_model, n_class, channels, window_size, activation, cnn_pooling)
    else:
        model = PoolClassifier(base_model, n_class, time_pooling, layer_pooling, layer)
    return model.to(device)
