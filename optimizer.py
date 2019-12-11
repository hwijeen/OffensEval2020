from transformers import AdamW, WarmupLinearSchedule


# TODO: training strategy
def build_optimizer_scheduler(model, lr, eps, warmup, weight_decay, train_step):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters()\
                    if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters()\
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr, eps, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup, train_step)
    return optimizer, scheduler


if __name__ == "__main__":
    pass
