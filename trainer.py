import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# In general, early stopping with dev set is needed
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
        loss = self.criterion(logits, batch.tgt).sum() / len(batch)
        return loss

    def _compute_entire_loss(self, data_iter):
        data_iter.repeat = False
        with torch.no_grad():
            loss = 0
            num_ex = 0
            for batch in data_iter:
                loss += self._compute_loss(batch) * len(batch) # silly
                num_ex += len(batch)
            loss = loss / num_ex
        data_iter.repeat = True
        return loss

    def train(self, train_step):
        for step, batch in enumerate(self.train_iter, 1):

            loss = self._compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if hasattr(self.model.encoder, 'lstm'):
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)


            if step % self.record_every == 0:
                test_loss = self._compute_entire_loss(self.test_iter)
                train_acc = self.evaluate(self.train_iter)
                test_acc = self.evaluate(self.test_iter)
                self.writer.add_scalar('Loss/train', loss.item(), step)
                self.writer.add_scalar('Loss/test', test_loss.item(), step)
                self.writer.add_scalar('Accuracy/train', train_acc, step)
                self.writer.add_scalar('Accuracy/test', test_acc, step)

            if step == train_step:
                self.writer.close()
                break

    def evaluate(self, data_iter):
        self.model.eval()
        data_iter.repeat = False
        predictions = []
        truth = []
        with torch.no_grad():
            for batch in data_iter:
                pred = self.model.inference(*batch.sent)
                predictions += pred.tolist()
                truth += batch.tgt.tolist()
        acc = calc_accuracy(predictions, truth)
        data_iter.repeat = True
        self.model.train()
        return acc
