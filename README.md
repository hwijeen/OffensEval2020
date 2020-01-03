This is an ongoing project for NLPDove🕊 in OffensEval2020.

## Data

Data for OffensEval2019 can be downloaded from the [official OLID dataset page](https://sites.google.com/site/offensevalsharedtask/olid).
The test data come in 3 files, one for each task. We merged the 3 files data into one so that it has the same format as the train data.

```bash
python merge_test.py # merge 3 test data into one file
```



## Quick start
Below is the command to train a model with some potentially important arguments.
For exhaustive list of arguments, read `train.py`.
```bash
python train.py --task a \
--demojize --lower_hashtag --segment_hashtag --textify_emoji --add_cap_sign \
--mention_limit 0 --punc_limit 0 \
--model bert --time_pooling max_avg --layer 12 \
--attention_probs_dropout_prob 0.1 --hidden_dropout_prob 0.3 \
--lr 0.00002 --weight_decay 0.0 --layer_decrease 1.0 --freeze_upto -1 --warmup_ratio 0.1 \
--batch_size 32 --train_step 700 --patience 20 --cuda 1 --note WRITENOTEHERE
```
To get a reasonable result, delete `--add_cap_sign`, change `--layer 11`, `freeze_upto 3`.



## Dependencies

