This is an ongoing project for NLPDove🕊 in OffensEval2020.

## Data

Data for OffensEval2019 can be downloaded from the [official OLID dataset page](https://sites.google.com/site/offensevalsharedtask/olid).
The test data come in 3 files, one for each task. We merged the 3 files data into one so that it has the same format as the train data.

```bash
python merge_test.py # merge 3 test data into one file
```



## Quick start

```bash
python train.py --task a --model bert --pooling avg --demojize --lower_hashtag --weight_decay 0.01 --warmup 1000 
```



## Dependencies

