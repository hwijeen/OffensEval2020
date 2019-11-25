import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import calc_acc


# TODO: early stopping
# TODO: running avg
# TODO: logging
# TODO: evaluation
class Trainer():
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
        #self.criterion = nn.BCEWithLogitsLoss()

    def _compute_loss(self, batch):
        logits = self.model(*batch.tweet)
        loss = self.criterion(logits, batch.label)
        return loss

    def _compute_entire_loss(self, data_iter):
        data_iter.repeat = False
        with torch.no_grad():
            losses = list(map(self._compute_loss, data_iter))
            loss = sum(losses) / len(losses)
        data_iter.repeat = True
        return loss

    def train(self, train_step):
        self.model.train()
        for step, batch in enumerate(self.train_iter, 1):
            loss = self._compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            #nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            if step % self.record_every == 0:
                val_loss = self._compute_entire_loss(self.val_iter)
                val_acc = self.evaluate(self.val_iter)
                self.record(loss, val_loss, step)
                print(f'At step: {step}')
                print(f'\tTrain loss: {loss.item()}')
                print(f'\tVal loss: {val_loss.item()}')
                print(f'\tval acc: {val_acc}')

                train_acc = self.evaluate(self.train_iter)
                print(f'\tTrain acc: {train_acc} - for debug!')

            if step == train_step:
                self.writer.close()
                break

    def record(self, loss, val_loss, step, train_acc=None, val_acc=None):
        self.writer.add_scalar('Loss/train', loss.item(), step)
        self.writer.add_scalar('Loss/val', val_loss.item(), step)
        if train_acc is not None:
            self.writer.add_scalar('Accuracy/train', train_acc, step)
        if val_acc is not None:
            self.writer.add_scalar('Accuracy/val', val_acc, step)

    def evaluate(self, data_iter):
        self.model.eval()
        data_iter.repeat = False
        predictions = []
        truth = []
        with torch.no_grad():
            for batch in data_iter:
                pred = self.model.predict(*batch.tweet)
                predictions += pred.tolist()
                truth += batch.label.tolist()
        acc = calc_acc(predictions, truth)
        data_iter.repeat = True
        self.model.train()
        return acc

def build_trainer(model, data, optimizer, scheduler, max_grad_norm, record_every,
                  exp_name):
    exp_name = exp_name
    trainer = Trainer(model, data.train_iter, data.val_iter, data.test_iter,
                      optimizer, scheduler, max_grad_norm, record_every,
                      exp_name)
    return trainer
