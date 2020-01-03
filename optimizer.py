import re
import logging

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule

logger = logging.getLogger(__name__)

def apply_wd(dt, model, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    for n, p in model.named_parameters():
        wd = weight_decay if not any(nd in n for nd in no_decay) else 0.0
        dt[n]['weight_decay'] = wd
    return dt, no_decay

def apply_disc_lr(dt, model, lr, layer_decrease):
    total_layer = model.model.config.num_hidden_layers
    layers_adjusted = set()
    for n, p in model.named_parameters():
        m = re.search('[\d]+|embedding', n)
        if m is None:
            continue
        elif m.group() == 'embedding':
            power = total_layer
        else:
            layer_idx = int(m.group())  # 0 to 11
            power = total_layer - layer_idx - 1
        lr_ = lr * (layer_decrease ** power)
        layers_adjusted.add(m.group())
        dt[n]['lr'] = lr_
    return dt, sorted(layers_adjusted)

def apply_layer_freeze(dt, model, freeze_upto):
    layers_adjusted = set()
    for n, p in model.named_parameters():
        m = re.search('[\d]+|embedding', n)
        if m is None:
            continue
        elif m.group() == 'embedding' or int(m.group()) <= freeze_upto:
            p.requires_grad = False
            dt[n]['params'] = p
            layers_adjusted.add(m.group())
    return dt, sorted(layers_adjusted)

def build_optimizer_scheduler(model, lr, betas, eps, warmup_ratio, weight_decay,
                              layer_decrease, freeze_upto, train_step):
    grouped_params = {n: {'params': p, 'lr': lr} for n, p in model.named_parameters()}
    if weight_decay != 0.0:
        grouped_params, no_decay = apply_wd(grouped_params, model, weight_decay)
        logger.info(f'Layer decay applied except for {no_decay}')
    if layer_decrease != 1.0:
        grouped_params, lr_changed = apply_disc_lr(grouped_params, model, lr, layer_decrease)
        logger.info(f'Discriminative fintuning - lr adjusted for {lr_changed}')
    if freeze_upto != -1:
        grouped_params, freezed = apply_layer_freeze(grouped_params, model, freeze_upto)
        logger.info(f'The following layers are freezed: {freezed}')

    optimizer_grouped_parameters= [param_dict for param_dict in grouped_params.values()]

    optimizer = AdamW(optimizer_grouped_parameters, lr, eps=eps, betas=betas, correct_bias=False)
    warmup = train_step * warmup_ratio
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, train_step)
    #scheduler = get_constant_schedule(optimizer)
    return optimizer, scheduler
