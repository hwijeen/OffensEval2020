import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# TODO: early stopping
# TODO: running avg
class Trainer():
    def __init__(self, model, train_iter, val_iter, test_iter, optimizer,
                 record_every=100, exp_name=None):
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        self.record_every = record_every
        self.writer = SummaryWriter(exp_name)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optimizer

    def _compute_loss(self, batch):
        logits = self.model(*batch.text)
        loss = self.criterion(logits, batch.label)
        return loss

    def _compute_entire_loss(self, data_iter):
        data_iter.repeat = False
        with torch.no_grad():
            loss = 0
            num_ex = 0
            for batch in data_iter:
                loss += self._compute_loss(batch) * len(batch)
                num_ex += len(batch)
            loss = loss / num_ex
        data_iter.repeat = True
        return loss

    def train(self, train_step):
        self.model.train()
        for step, batch in enumerate(self.train_iter, 1):
            loss = self._compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % self.record_every == 0:
                val_loss = self._compute_entire_loss(self.val_iter)
                self.record(loss, val_loss)

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

    #def evaluate(self, data_iter):
    #    self.model.eval()
    #    data_iter.repeat = False
    #    predictions = []
    #    truth = []
    #    with torch.no_grad():
    #        for batch in data_iter:
    #            pred = self.model.inference(*batch.sent)
    #            predictions += pred.tolist()
    #            truth += batch.tgt.tolist()
    #    acc = calc_accuracy(predictions, truth)
    #    data_iter.repeat = True
    #    self.model.train()
    #    return acc
