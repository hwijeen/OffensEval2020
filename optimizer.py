from transformers import AdamW, WarmupLinearSchedule


# TODO: training strategy
def build_optimizer_scheduler(model, lr, eps, warmup, train_step):
    # TODO: below is suggested in McCormick's blog
    #param_optimizer = list(model.named_parameters())
    #no_decay = ['bias', 'gamma', 'beta']
    #optimizer_grouped_parameters = [
    #    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.01},
    #    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
    #     'weight_decay_rate': 0.0}
    #]
    #optimizer = AdamW(optimizer_grouped_parameters, lr, eps, correct_bias=False)
    optimizer = AdamW(model.parameters(), lr, eps, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup, train_step)
    return optimizer, scheduler


if __name__ == "__main__":
    pass
