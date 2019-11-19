import argparse
import logging

import torch

from dataloading import build_data
from preprocessing import build_preprocess, build_tokenizer
from model import build_model
from trainer import build_trainer
from optimizer import build_optimizer_scheduler


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    data = parser.add_argument_group('Data')
    data.add_argument('--task', choices=['a', 'b', 'c'], default='a')
    data.add_argument('--train_path', default='../data/olid-training-v1.0.tsv')
    data.add_argument('--test_path', default='../data/testset-level.tsv')

    preprocess = parser.add_argument_group('Preprocessing options')
    #preprocess.add_argument('--capitalize')
    preprocess.add_argument('--punctuation')
    preprocess.add_argument('--no_demojize', action='store_true')
    preprocess.add_argument('--emoji_min_freq', type=int, default=10)
    preprocess.add_argument('--hashtag_min_freq', type=int, default=1)
    preprocess.add_argument('--tokenize', default='bert')

    model = parser.add_argument_group('Model options')
    model.add_argument('--model', choices=['bert', 'bert_avg'], default='bert')

    optimizer_scheduler = parser.add_argument_group(('Optimizer and scheduler options'))
    optimizer_scheduler.add_argument('--lr', type=float, default=0.001)
    optimizer_scheduler.add_argument('--beta1', type=float, default=0.9)
    optimizer_scheduler.add_argument('--beta2', type=float, default=0.999)
    optimizer_scheduler.add_argument('--warmup', type=int, default=100)

    training = parser.add_argument_group('Training options')
    training.add_argument('--batch_size', type=int, default=32)
    training.add_argument('--cuda', action='store_true', default=True)
    training.add_argument('--train_step', type=int, default=100000)
    training.add_argument('--record_every', type=int, default=100)

    args = parser.parse_args()
    # TODO: clean these hacks..
    args.device = torch.device('cuda') if args.cuda else torch.device('cpu')
    args.test_path = args.test_path.replace('level', f'level{args.task}')
    return args

def generate_exp_name(args):
    return

if __name__ == "__main__":

    args = parse_args()
    exp_name = generate_exp_name(args)
    preprocessing = build_preprocess(no_demojize=args.no_demojize)
    tokenizer = build_tokenizer(args.model)
    olid_data = build_data(model=args.model,
                           train_path=args.train_path,
                           test_path=args.test_path,
                           task=args.task,
                           preprocessing=preprocessing,
                           tokenizer=tokenizer,
                           batch_size=args.batch_size,
                           device=args.device)
    model = build_model(task=args.task,
                        model=args.model,
                        device=args.device)
    optimizer, scheduler = build_optimizer_scheduler(model=model,
                                                     lr=args.lr,
                                                     eps=(args.beta1, args.beta2),
                                                     warmup=args.warmup,
                                                     train_step=args.train_step)
    trainer = build_trainer(model=model,
                            data=olid_data,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            record_every=args.record_every,
                            exp_name=exp_name)

    logger.info(f'Training logs are in {exp_name}')
    logger.info(f'Preprocessing options')
    logger.info(f'Number of vocab and data size')
    trainer.train(args.train_step)
