import os
import logging

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import *

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model, patience, savedir, delta=0, mode='max', verbose=False):
        self.model = model
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.savedir = os.path.join(savedir, 'best_model_checkpoint.pt')
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
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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
            logger.info(f'Best score on validation improved ({self.prev_best_score:.6f} -->'
                  f'{curr_score:.6f}). Checkpoint model saved.')
        torch.save(self.model.state_dict(), self.savedir)
        self.prev_best_score = curr_score

    def delete_checkpoint(self):
        if os.path.exists(self.savedir):
            os.remove(self.savedir)


class Trainer:
    def __init__(self, model, train_iter, val_iter, optimizer, scheduler,
                 max_grad_norm, patience, exp_name, record_every=100,
                 verbose=True, test_iter=None):
        self.model = model
        self.train_iter = train_iter
        self.val_iter = val_iter
        self.test_iter = test_iter
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.record_every = record_every
        self.criterion = nn.CrossEntropyLoss()
        self.exp_dir = rename_expname(exp_name)
        self.early_stopper = EarlyStopping(model, patience, self.exp_dir, verbose=verbose)
        self.writer = SummaryWriter(self.exp_dir)
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
            self.model.train()
            loss = self.compute_loss(batch)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()

            if step % self.record_every == 0:
                val_loss = self.compute_entire_loss(self.val_iter)
                val_metrics = self.evaluate(self.val_iter)
                train_metrics = self.evaluate(self.train_iter) # optional
                self.record('val', step, *val_metrics, loss=val_loss)
                self.record('train', step, *train_metrics, loss=loss)
                if self.test_iter is not None:
                    test_loss = self.compute_entire_loss(self.test_iter)
                    test_metrics = self.evaluate(self.test_iter)
                    self.record('test', step, *test_metrics, loss=test_loss)
                self.writer.add_scalar('Learning_rate', self.scheduler.get_lr()[0], step)

                if self.verbose:
                    print(f'At step: {step}')
                    self.report('train', *train_metrics, loss=loss)
                    self.report('val', *val_metrics, loss=val_loss)
                    self.report('test', *test_metrics)

                val_f1 = val_metrics[0]
                self.early_stopper(step, val_f1)
                if self.early_stopper.early_stop:
                    logger.info(f'..... Early stopping patience reached at step {step}, terminating training .....')
                    return self.finish_training()

            if step == train_step:
                logger.info(f'\n..... Max train step({train_step}) reached, terminating training .....\n')
                return self.finish_training()

    def finish_training(self):
        self.model.load_state_dict(torch.load(self.early_stopper.savedir))
        summary = self.summarize_training()
        self.writer.add_text('Summary', summary)
        self.early_stopper.delete_checkpoint()
        self.writer.close()
        return self.model, summary

    def summarize_training(self):
        summary = f'Best model was found at step: {self.early_stopper.best_step}\n'
        summary += 'On validation data:\n'
        f1, prec, rec, acc = self.evaluate(self.val_iter)
        summary += f'accuracy-{acc:.4f}, precision-{prec:.4f}, recall-{rec:.4f}, f1-{f1:.4f}\n'
        if self.test_iter is not None:
            summary += 'On test data:\n'
            f1, prec, rec, acc = self.evaluate(self.test_iter)
        summary += f'accuracy-{acc:.4f}, precision-{prec:.4f}, recall-{rec:.4f}, f1-{f1:.4f}'
        return summary

    def record(self, kind, step, f1, prec, rec, acc, loss=None):
        assert kind in {'train', 'val', 'test'}
        self.writer.add_scalar(f'F1/{kind}', f1, step)
        self.writer.add_scalar(f'Precision/{kind}', prec, step)
        self.writer.add_scalar(f'Recall/{kind}', rec, step)
        self.writer.add_scalar(f'Acc/{kind}', acc, step)
        if loss is not None:
            self.writer.add_scalar(f'Loss/{kind}', loss.item(), step) # batch loss

    def report(self, kind, f1, prec, rec, acc, loss=None):
        assert kind in {'train', 'val', 'test'}
        if loss is not None:
            print(f'\t{kind} loss: {loss.item():.6f}')
        print(f'\t{kind} F1: {f1:.6f}')

    def evaluate(self, data_iter):
        self.model.eval()
        data_iter.repeat = False
        predictions, golds = [], []
        with torch.no_grad():
            for batch in data_iter:
                pred = self.model.predict(*batch.tweet)
                predictions += pred.tolist()
                golds += batch.label.tolist()
        f1 = calc_f1(predictions, golds)
        prec = calc_prec(predictions, golds)
        rec = calc_rec(predictions, golds)
        acc = calc_acc(predictions, golds)
        data_iter.repeat = True
        return f1, prec, rec, acc


# TODO: make verbose an option
def build_trainer(model, data, optimizer, scheduler, max_grad_norm,
                  record_every, patience, exp_name):
    trainer = Trainer(model, data.train_iter, data.val_iter, optimizer,
                      scheduler, max_grad_norm, patience, exp_name,
                      record_every, verbose=True, test_iter=data.test_iter)
    return trainer
