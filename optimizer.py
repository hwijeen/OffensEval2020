from transformers import AdamW, WarmupLinearSchedule


# TODO: training strategy
def build_optimizer_scheduler(model, lr, eps, warmup, train_step):
    optimizer = AdamW(model.parameters(), lr, eps, correct_bias=False)
    scheduler = WarmupLinearSchedule(optimizer, warmup, train_step)
    return optimizer, scheduler


if __name__ == "__main__":
    pass
