import re
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

# TODO: training strategy
def build_optimizer_scheduler(model, lr, betas, eps, warmup_proportion, weight_decay,
                              layer_decrease, train_step):

    def set_layer_lr(param_name):
        m = re.search('[\d]', param_name)
        return lr * layer_decrease ** (model.model.config.num_hidden_layers - int(m.group())) if m else lr

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters= []
    for n, p in model.named_parameters():
        wd = weight_decay if not any(nd in n for nd in no_decay) else 0.0
        lr_ = set_layer_lr(n)
        param_setting = {'params': p, 'weight_decay': wd, 'lr': lr_}
        optimizer_grouped_parameters.append(param_setting)

    optimizer = AdamW(optimizer_grouped_parameters, lr, eps=eps, betas=betas, correct_bias=False)
    warmup = train_step * warmup_proportion
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, train_step)
    #scheduler = get_constant_schedule(optimizer)
    return optimizer, scheduler
