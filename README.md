This is an ongoing project for NLPDove🕊 in OffensEval2020.

## Data

Data for OffensEval2019 can be downloaded from the [official OLID dataset page](https://sites.google.com/site/offensevalsharedtask/olid).
The test data come in 3 files, one for each task. We merged the 3 files data into one so that it has the same format as the train data.

```bash
python merge_test.py # merge 3 test data into one file
cut -d ',' -f6- ../resources/training.1600000.processed.noemoticon.csv | sed -e 's/^"//' -e 's/"$//' > ../resources/tweet_corpus_raw.txt # prepare corpus for LM-finetune
```



## Quick start

```bash
#python run_lm_finetune.py ---demojize --lower_hashtag --replace_user -mlm --do_train --block_size 20 --do_lower_case --output_dir finetuned_by_examples --model_name_or_path finetuned_by_examples/checkpoint-50000 --overwrite_output_dir --num_train_epochs 2.5 --note resuming 
nohup python run_lm_finetune.py --demojize --lower_hashtag --textify_emoji --segment_hashtag --replace_user\
                                --warmup_steps_proportion 0.0
                                --mlm --do_lower_case --do_train  --num_train_epochs 30 --block_size 1\
                                --output_dir finetuned_with_added_tokens --overwrite_output_dir\
                                --note finetuned_with_added_tokens > ../finetuned_with_added_tokens &
python train.py --task a --model finetuned_by_examples/checkpoint-74100  --pooling avg --demojize --lower_hashtag --weight_decay 0.01 --warmup 1000 
```



## Dependencies

