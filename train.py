import argparse
import logging
from setproctitle import setproctitle

import torch

from dataloading import build_data
from model import build_model
from trainer import build_trainer
from utils import write_result_to_file, write_summary_to_file
from optimizer import build_optimizer_scheduler
from preprocessing import build_preprocess, build_tokenizer

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: inference.py with test data
# TODO: defaults to None when no testdata available
def parse_args():
    parser = argparse.ArgumentParser()
    data = parser.add_argument_group('Data')
    data.add_argument('--task', choices=['a', 'b', 'c'], default='a')
    data.add_argument('--train_path', default='../data/olid-training-v1.0.tsv')
    data.add_argument('--test_path', default='../data/olid-test-v1.0.tsv')

    preprocess = parser.add_argument_group('Preprocessing options')
    preprocess.add_argument('--punctuation') # not implemented
    preprocess.add_argument('--demojize', action='store_true')
    preprocess.add_argument('--emoji_min_freq', type=int, default=10)
    preprocess.add_argument('--lower_hashtag', action='store_true')
    preprocess.add_argument('--hashtag_min_freq', type=int, default=10)
    preprocess.add_argument('--add_cap_sign', action='store_true')
    preprocess.add_argument('--mention_limit', type=int, default=3)
    preprocess.add_argument('--punc_limit', type=int, default=3)
    preprocess.add_argument('--tokenize', default='bert')
    preprocess.add_argument('--segment_hashtag', action='store_true')

    model = parser.add_argument_group('Model options')
    model.add_argument('--model', choices=['bert', 'roberta', 'xlm', 'xlnet'], default='bert')
    model.add_argument('--pooling', choices=['cls', 'avg'], default='avg')
    model.add_argument('--attention_probs_dropout_prob', type=float, default=0.3)
    model.add_argument('--hidden_dropout_prob', type=float, default=0.1)

    optimizer_scheduler = parser.add_argument_group('Optimizer and scheduler options')
    optimizer_scheduler.add_argument('--lr', type=float, default=0.00005)
    optimizer_scheduler.add_argument('--beta1', type=float, default=0.9)
    optimizer_scheduler.add_argument('--beta2', type=float, default=0.999)
    optimizer_scheduler.add_argument('--warmup', type=int, default=1000)
    optimizer_scheduler.add_argument('--max_grad_norm', type=float, default=1.0)
    optimizer_scheduler.add_argument('--weight_decay', type=float, default=0.0)
    optimizer_scheduler.add_argument('--layer_decrease', type=float, default=1.0)

    training = parser.add_argument_group('Training options')
    training.add_argument('--batch_size', type=int, default=32)
    training.add_argument('--cuda', type=int, default=0)
    training.add_argument('--train_step', type=int, default=700)
    training.add_argument('--record_every', type=int, default=10)
    training.add_argument('--patience', type=int, default=10)
    training.add_argument('--note', type=str, default='')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    # TODO: clean these hacks..
    args.device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    if args.debug:
        args.train_path = '../data/debug_train.tsv'
        print('Debug mode!!!!')
    return args

def generate_exp_name(args):
    model = f'model_{args.model}'
    pooling = f'pooling_{args.pooling}'
    lr = f'lr_{args.lr}'
    task = f'task_{args.task}'
    exp_name = '_'.join([model, pooling, lr, task, args.note])
    return exp_name

# TODO: save args for reproducible exp
if __name__ == "__main__":
    args = parse_args()
    exp_name = generate_exp_name(args)
    setproctitle(exp_name)
    preprocess = build_preprocess(demojize=args.demojize,
                                  mention_limit=args.mention_limit,
                                  punc_limit=args.punc_limit,
                                  lower_hashtag=args.lower_hashtag,
                                  add_cap_sign=args.add_cap_sign,
                                  segment_hashtag=args.segment_hashtag)
    tokenizer = build_tokenizer(model=args.model,
                                emoji_min_freq=args.emoji_min_freq,
                                hashtag_min_freq=args.hashtag_min_freq,
                                add_cap_sign=args.add_cap_sign,
                                preprocess=preprocess)
    olid_data = build_data(model=args.model,
                           train_path=args.train_path,
                           task=args.task,
                           preprocessing=None,
                           tokenizer=tokenizer,
                           batch_size=args.batch_size,
                           device=args.device,
                           test_path=args.test_path)
    model = build_model(task=args.task,
                        model=args.model,
                        pooling=args.pooling,
                        new_num_tokens=len(tokenizer),
                        hidden_dropout_prob=args.hidden_dropout_prob,
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        device=args.device)
    optimizer, scheduler = build_optimizer_scheduler(model=model,
                                                     lr=args.lr,
                                                     eps=(args.beta1, args.beta2),
                                                     warmup=args.warmup,
                                                     weight_decay=args.weight_decay,
                                                     layer_decrease=args.layer_decrease,
                                                     train_step=args.train_step)
    trainer = build_trainer(model=model,
                            data=olid_data,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            max_grad_norm=args.max_grad_norm,
                            patience=args.patience,
                            record_every=args.record_every,
                            exp_name=exp_name)

    logger.info(f'Training logs are in {exp_name}')
    logger.info(f'Preprocessing options')
    logger.info(f'Number of vocab and data size')
    trained_model, summary = trainer.train(args.train_step)

    pred_file_name = f'runs/{exp_name}/prediction.tsv'
    summary_file_name = f'runs/{exp_name}/summary.txt'
    write_result_to_file(trained_model, trainer.test_iter, tokenizer,
                         args, pred_file_name)
    write_summary_to_file(summary, args, summary_file_name)

    print('\n******************* Training summary *******************')
    print(f'exp_name: {exp_name}', end='\n\n')
    print(summary)
    print('********************************************************')

