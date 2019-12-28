import os
import os
import argparse
import logging

import torch
from setproctitle import setproctitle

from dataloading import build_data
from model import build_model
from trainer import build_trainer
from utils import *
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
    preprocess.add_argument('--demojize', action='store_true')
    preprocess.add_argument('--lower_hashtag', action='store_true')
    preprocess.add_argument('--segment_hashtag', action='store_true')
    preprocess.add_argument('--textify_emoji', action='store_true')
    preprocess.add_argument('--add_cap_sign', action='store_true')
    preprocess.add_argument('--emoji_min_freq', type=int, default=0)
    preprocess.add_argument('--hashtag_min_freq', type=int, default=0)
    preprocess.add_argument('--mention_limit', type=int, default=3)
    preprocess.add_argument('--punc_limit', type=int, default=3)

    model = parser.add_argument_group('Model options')
    model.add_argument('--model', choices=['bert', 'roberta', 'xlm', 'xlnet'], default='bert')
    model.add_argument('--time_pooling', choices=['cls', 'avg', 'max', 'max_avg', 'cnn'], default='max_avg')
    model.add_argument('--layer_pooling', choices=['avg', 'weight', 'max', 'cat'], default='cat')
    model.add_argument('--layer', type=int, choices=range(1, 13), nargs='+', default=[12])
    model.add_argument('--attention_probs_dropout_prob', type=float, default=0.1)
    model.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    model.add_argument('--window_size', type=int, nargs='+', default=[3])
    model.add_argument('--channels', type=int, default=1)
    model.add_argument('--cnn_pooling', choices=['avg', 'max', 'max_avg'], default='max')
    model.add_argument('--activation', choices=['relu', 'lrelu', 'glu'], default='relu')
    model.add_argument('--wind_per_layer', action='store_true')

    optimizer_scheduler = parser.add_argument_group('Optimizer and scheduler options')
    optimizer_scheduler.add_argument('--lr', type=float, default=0.00002)
    optimizer_scheduler.add_argument('--beta1', type=float, default=0.9)
    optimizer_scheduler.add_argument('--beta2', type=float, default=0.999)
    optimizer_scheduler.add_argument('--eps', type=float, default=1e-6)
    optimizer_scheduler.add_argument('--warmup_ratio', type=float, default=0.1)
    optimizer_scheduler.add_argument('--max_grad_norm', type=float, default=1.0)
    optimizer_scheduler.add_argument('--weight_decay', type=float, default=0.0)
    optimizer_scheduler.add_argument('--layer_decrease', type=float, default=1.0)

    training = parser.add_argument_group('Training options')
    training.add_argument('--batch_size', type=int, default=32)
    training.add_argument('--train_step', type=int, default=700)
    training.add_argument('--record_every', type=int, default=10)
    training.add_argument('--patience', type=int, default=20)
    training.add_argument('--cuda', type=int, default=0)
    training.add_argument('--note', type=str, default='')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    # TODO: clean these hacks..
    args.device = torch.device(f'cuda:{args.cuda}') if torch.cuda.is_available() else torch.device('cpu')
    if args.debug:
        args.train_path = '../data/debug_train.tsv'
        print('Debug mode!!!!')
    return args

def generate_exp_name(args, preprocessing=True, modeling=True, optim_schedule=True, training=False):
    to_include = [f'{args.task}'.upper()]

    if preprocessing:
        demojize = f'Demoji:{str(args.demojize)}'
        lower_hashtag = f'LowHash:{str(args.lower_hashtag)}'
        add_cap_sign = f'CapSign:{str(args.add_cap_sign)}'
        segment_hashtag = f'SegHash:{str(args.segment_hashtag)}'
        textify_emoji = f'TextEmoji:{str(args.textify_emoji)}'
        to_include.append('_'.join([demojize, lower_hashtag, add_cap_sign, segment_hashtag, textify_emoji]))

    if modeling:
        model = f'{args.model}'.replace('/', '_').upper() # for `some/checkpoint`
        time_pool = f'TimePool:{args.time_pooling}'
        layer_pool = f'LayerPool:{args.layer_pooling}'
        layer = 'Layer:' + ','.join(map(str, args.layer))
        attn_dropout = f'AttnDrop:{args.attention_probs_dropout_prob}'
        hidden_dropout = f'HidDrop:{args.hidden_dropout_prob}'
        window_size = f'Wind:' + ','.join(map(str, args.window_size))
        channels = f'Chan:{args.channels}'
        cnn_pooling = f'CnnPool:{args.cnn_pooling}'
        activation = f'Actv:{args.activation}'
        to_include.append('_'.join([model, time_pool, layer_pool, layer, attn_dropout, hidden_dropout,
                                    window_size, channels, cnn_pooling, activation]))

    if optim_schedule:
        lr = f'Lr:{args.lr}'
        warmup_ratio = f'WarmUp:{args.warmup_ratio}'
        decay = f'Decay:{args.weight_decay}'
        layer_decrease = f'LayerDec:{args.layer_decrease}'
        to_include.append('_'.join([lr, warmup_ratio, decay, layer_decrease]))

    if training:
        batch = f'Batch:{args.batch_size}'
        train = f'Train:{args.train_step}'
        patience = f'Patience:{args.patience}'
        to_include.append('_'.join([batch, train, patience]))

    to_include.append('_'.join([args.note]))
    return '_'.join(to_include)

if __name__ == "__main__":
    args = parse_args()
    exp_name = generate_exp_name(args)
    setproctitle(args.note)
    preprocess = build_preprocess(demojize=args.demojize,
                                  mention_limit=args.mention_limit,
                                  punc_limit=args.punc_limit,
                                  lower_hashtag=args.lower_hashtag,
                                  add_cap_sign=args.add_cap_sign,
                                  segment_hashtag=args.segment_hashtag,
                                  textify_emoji=args.textify_emoji)
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
                        time_pooling=args.time_pooling,
                        layer_pooling=args.layer_pooling,
                        layer=args.layer,
                        window_size=args.window_size,
                        channels=args.channels,
                        cnn_pooling=args.cnn_pooling,
                        activation=args.activation,
                        wind_per_layer = args.wind_per_layer,
                        new_num_tokens=len(tokenizer),
                        hidden_dropout_prob=args.hidden_dropout_prob,
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        device=args.device)
    optimizer, scheduler = build_optimizer_scheduler(model=model,
                                                     lr=args.lr,
                                                     betas=(args.beta1, args.beta2),
                                                     eps=args.eps,
                                                     warmup_ratio=args.warmup_ratio,
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
    trained_model, summary = trainer.train(args.train_step)

    best_model_file = os.path.join(trainer.exp_dir, 'best_model.pt')
    pred_file = os.path.join(trainer.exp_dir, 'prediction.tsv')
    summary_file = os.path.join(trainer.exp_dir, 'summary.txt')
    args_file = os.path.join(trainer.exp_dir, 'args.bin')
    save_model(trained_model, best_model_file)
    save_tokenizer(tokenizer, trainer.exp_dir)
    write_pred_to_file(trained_model, trainer.test_iter, tokenizer, pred_file)
    write_args_to_file(args, args_file)
    write_summary_to_file(summary, summary_file)

    print('\n******************* Training summary *******************')
    print(summary, end='\n\n')
    print('Best model, tokenizer, prediction, args, summary are saved')
    print(f'Tensorboard exp_name: {exp_name}')
    print('********************************************************')

