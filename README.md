This is an ongoing project for Transferrable Data Selection.

## Data

OffensEval2019: [Official OLID dataset page](https://sites.google.com/site/offensevalsharedtask/olid).
Data for OffensEval2020: upon request.

```bash
python merge_test.py # OffensEval2019 - merge 3 test data into one file
```



## Quick start
Below is the command to train a model with some potentially important arguments.
For exhaustive list of arguments, read `train.py`.
```bash
python train.py \
--train_path ../data/olid/da/offenseval-da-training-v1-train.tsv \
--test_path ../data/olid/da/offenseval-da-training-v1-test.tsv \
--demojize --lower_hashtag --segment_hashtag --textify_emoji --add_cap_sign \
--mention_limit 0 --punc_limit 0 \
--model bert --time_pooling max_avg --layer 12 \
--attention_probs_dropout_prob 0.1 --hidden_dropout_prob 0.3 \
--lr 0.00002 --weight_decay 0.0 --layer_decrease 1.0 --freeze_upto -1 --warmup_ratio 0.1 \
--batch_size 32 --train_step 700 --patience 20 --cuda 1 --note WRITENOTEHERE
```
To get a reasonable result, delete `--add_cap_sign`, change `--layer 11`, `freeze_upto 3`.



## Dependencies

