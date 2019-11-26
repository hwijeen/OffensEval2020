import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import calc_acc, calc_f1


# TODO: early stopping
# TODO: logging
# TODO: evaluation
class Trainer:
    def __init__(self, model, train_iter, val_iter, test_iter, optimizer,
                 scheduler, max_grad_norm, record_every=100, exp_name=None):
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.record_every = record_every
        self.writer = SummaryWriter('runs/'+exp_name)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss(self, batch):
        logits = self.model(*batch.tweet)
        loss = self.criterion(logits, batch.label)
        return loss

    def compute_entire_loss(self, data_iter):
        data_iter.repeat = False
        with torch.no_grad():
            losses = list(map(self.compute_loss, data_iter))
            loss = sum(losses) / len(losses) # TODO: not exactly accurate
        data_iter.repeat = True
        return loss

    def train(self, train_step):
        for step, batch in enumerate(self.train_iter, 1):
            self.model.train()
            loss = self.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            if step % self.record_every == 0:
                val_loss = self.compute_entire_loss(self.val_iter)
                val_acc, val_f1 = self.evaluate(self.val_iter)
                train_acc, train_f1 = self.evaluate(self.train_iter) # optional
                self.record(step, loss, val_loss, val_acc, val_f1, train_acc, train_f1)
                self.report(step, loss, val_loss, val_acc, val_f1, train_acc, train_f1)

            if step == train_step:
                test_acc, test_f1 = self.evaluate(self.test_iter)
                print('*'*60)
                print(f'Train finished with test acc:{test_acc}, test_f1:{test_f1}')
                print('*'*60)
                self.writer.close()
                return self.model

    def record(self, step, loss, val_loss, val_acc, val_f1, train_acc=None, train_f1=None):
        self.writer.add_scalar('Loss/train', loss.item(), step)
        self.writer.add_scalar('Loss/val', val_loss.item(), step)
        self.writer.add_scalar('Acc/val', val_acc, step)
        self.writer.add_scalar('F1/val', val_f1, step)
        if train_acc is not None:
            self.writer.add_scalar('Acc/train', train_acc, step)
        if train_f1 is not None:
            self.writer.add_scalar('F1/train', train_f1, step)

    def report(self, step, loss, val_loss, val_acc, val_f1, train_acc=None, train_f1=None):
        print(f'At step: {step}')
        print(f'\tTrain loss: {loss.item():.6f}')
        print(f'\tVal loss: {val_loss.item():.6f}')
        print(f'\tVal acc: {val_acc:.2f}')
        print(f'\tVal F1: {val_f1:.2f}')
        if train_acc is not None:
            print(f'\tTrain acc: {train_acc:.2f} - for debug!')
        if train_f1 is not None:
            print(f'\tTrain f1: {train_f1:.2f} - for debug!')

    def evaluate(self, data_iter):
        self.model.eval()
        data_iter.repeat = False
        predictions, golds = [], []
        with torch.no_grad():
            for batch in data_iter:
                pred = self.model.predict(*batch.tweet)
                predictions += pred.tolist()
                golds += batch.label.tolist()
        acc = calc_acc(predictions, golds)
        f1 = calc_f1(predictions, golds)
        data_iter.repeat = True
        return acc, f1

# TODO: is this necessary?
def build_trainer(model, data, optimizer, scheduler, max_grad_norm, record_every,
                  exp_name):
    exp_name = exp_name
    trainer = Trainer(model, data.train_iter, data.val_iter, data.test_iter,
                      optimizer, scheduler, max_grad_norm, record_every,
                      exp_name)
    return trainer
