import os
import argparse
import logging

import torch
from setproctitle import setproctitle

from dataloading import build_data
from model import build_model
from trainer import evaluate
from utils import *
from optimizer import build_optimizer_scheduler
from preprocessing import build_preprocess, build_tokenizer

# torch.manual_seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_note')
    parser.add_argument('--test_path', default='../data/olid/da/offenseval-da-training-v1-test.tsv')

    # Load from saved args
    args = parser.parse_args()
    saved_args = load_args_from_file(args.exp_note)
    saved_args.exp_note = args.exp_note
    saved_args.test_path = args.test_path
    return saved_args

def generate_exp_name(args):
    exp_note = args.exp_note
    test_lang = args.test_path.split('/')[3]
    return exp_note + 'test_on_' + test_lang

if __name__ == "__main__":
    args = parse_args()
    exp_name = generate_exp_name(args)
    setproctitle(args.exp_note)
    preprocess = build_preprocess(demojize=args.demojize,
                                  textify_emoji=args.textify_emoji,
                                  mention_limit=args.mention_limit,
                                  punc_limit=args.punc_limit,
                                  lower_hashtag=args.lower_hashtag,
                                  segment_hashtag=args.segment_hashtag,
                                  add_cap_sign=args.add_cap_sign)
    tokenizer = build_tokenizer(model=args.model,
                                add_cap_sign=args.add_cap_sign,
                                textify_emoji=args.textify_emoji,
                                segment_hashtag=args.segment_hashtag,
                                preprocess=preprocess)
    preproc = lambda x: x[:509]
    olid_data = build_data(model=args.model,
                           preprocessing=preproc,
                           tokenizer=tokenizer,
                           batch_size=args.batch_size,
                           device=args.device,
                           train_path=None,
                           test_path=args.test_path)
    model = build_model(model=args.model,
                        time_pooling=args.time_pooling,
                        layer_pooling=args.layer_pooling,
                        layer=args.layer,
                        new_num_tokens=len(tokenizer),
                        hidden_dropout_prob=args.hidden_dropout_prob,
                        attention_probs_dropout_prob=args.attention_probs_dropout_prob,
                        device=args.device)
    model = load_model(model, args.exp_note)
    optimizer, scheduler = build_optimizer_scheduler(model=model,
                                                     lr=args.lr,
                                                     betas=(args.beta1, args.beta2),
                                                     eps=args.eps,
                                                     warmup_ratio=args.warmup_ratio,
                                                     weight_decay=args.weight_decay,
                                                     layer_decrease=args.layer_decrease,
                                                     freeze_upto=args.freeze_upto,
                                                     train_step=args.train_step)
    f1, prec, rec, acc = evaluate(model, olid_data.test_iter)
    print()
    print('*'*80)
    print(f'Model loaded from: {args.exp_note}')
    print(f'Tested on: {args.test_path}')
    print(f'F1: {f1}')
    print(f'recall: {rec}')
    print(f'accuracy: {acc}')
    print('*'*80)
    print()

    pred_file = os.path.join('preds/', exp_name + '_prediction.tsv')
    write_pred_to_file(model, olid_data.test_iter, tokenizer, pred_file)
