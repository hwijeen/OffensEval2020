import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import calc_acc, calc_f1


# TODO: logging
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model, patience, savedir, delta=0, mode='max', verbose=False):
        self.model = model
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.savedir = os.path.join(savedir, 'best_model.pt')
        self.verbose = verbose
        self.counter = 0
        self.best_step = None
        self.best_score = None
        self.prev_best_score = float('inf')
        self.early_stop = False

    def __call__(self, step, val_score):

        score = -val_score if self.mode == 'min' else val_score

        if self.best_score is None:
            self.best_step = step
            self.best_score = score
            self.save_checkpoint()
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_step = step
            self.best_score = score
            self.save_checkpoint()
            self.counter = 0

    def save_checkpoint(self):
        '''Saves model when validation score improves.'''
        curr_score = -self.best_score if self.mode == 'min' else self.best_score
        if self.verbose:
            print(f'Best score on validation improved ({self.prev_best_score:.6f} -->'
                  f'{curr_score:.6f}).  Saving model ...')
        torch.save(self.model.state_dict(), self.savedir)
        self.prev_best_score = curr_score



class Trainer:
    def __init__(self, model, train_iter, val_iter, test_iter, optimizer,
                 scheduler, max_grad_norm, patience, record_every=100,
                 exp_name=None, verbose=True):
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.record_every = record_every
        self.criterion = nn.CrossEntropyLoss()
        exp_dir = os.path.join('runs', exp_name)
        self.early_stopper = EarlyStopping(model, patience, exp_dir, verbose=verbose)
        self.writer = SummaryWriter(exp_dir)
        self.verbose = verbose

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
            input()
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
                test_acc, test_f1 = self.evaluate(self.test_iter)
                self.record(step, loss, val_loss, val_acc, val_f1,
                            train_acc, train_f1, test_acc, test_f1)
                if self.verbose:
                    self.report(step, loss, val_loss, val_acc, val_f1,
                                train_acc, train_f1)
                self.early_stopper(step, val_f1)
                if self.early_stopper.early_stop:
                    print(f'Early stopping at {step} step.')
                    self.model.load_state_dict(torch.load(self.early_stopper.savedir))
                    step = train_step # terminates training in the next line

            if step == train_step:
                test_acc, test_f1 = self.evaluate(self.test_iter)
                print('*'*60)
                print(f'Train finished at {self.early_stopper.best_step} step,')
                print(f'Test acc:{test_acc}, Test_f1:{test_f1}')
                print('*'*60)
                self.writer.close()
                return self.model

    def record(self, step, loss, val_loss, val_acc, val_f1,
               train_acc=None, train_f1=None, test_acc=None, test_f1=None):
        self.writer.add_scalar('Loss/train', loss.item(), step)
        self.writer.add_scalar('Loss/val', val_loss.item(), step)
        self.writer.add_scalar('Acc/val', val_acc, step)
        self.writer.add_scalar('F1/val', val_f1, step)
        if train_acc is not None:
            self.writer.add_scalar('Acc/train', train_acc, step)
        if train_f1 is not None:
            self.writer.add_scalar('F1/train', train_f1, step)
        if test_acc is not None:
            self.writer.add_scalar('Acc/test', test_acc, step)
        if test_f1 is not None:
            self.writer.add_scalar('F1/test', test_f1, step)

    def report(self, step, loss, val_loss, val_acc, val_f1, train_acc=None,
               train_f1=None):
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
def build_trainer(model, data, optimizer, scheduler, max_grad_norm,
                  record_every, patience, exp_name):
    exp_name = exp_name
    trainer = Trainer(model, data.train_iter, data.val_iter, data.test_iter,
                      optimizer, scheduler, max_grad_norm, patience,
                      record_every, exp_name)
    return trainer
