#!/bin/bash
train_data=(../data/olid/ar/offenseval-ar-training-v1-train.tsv\
            ../data/olid/da/offenseval-da-training-v1-train.tsv\
            ../data/olid/el/offenseval-greek-training-v1-train.tsv\
            ../data/olid/en/olid-training-v1.0-train.tsv\
            ../data/olid/tr/offenseval-tr-training-v1-train.tsv)

test_data=(../data/olid/ar/offenseval-ar-training-v1-test.tsv\
           ../data/olid/da/offenseval-da-training-v1-test.tsv\
           ../data/olid/el/offenseval-greek-training-v1-test.tsv\
           ../data/olid/en/olid-training-v1.0-test.tsv\
           ../data/olid/tr/offenseval-tr-training-v1-test.tsv)

notes=(mBERT_ar mBERT_da mBERT_el mBERT_en mBERT_tr)

for ((i=0; i<${#test_data[@]}; i++));do
    tr=${train_data[$i]}
    tst=${test_data[$i]}
    note=${notes[$i]}
    echo train on $tr and test on $tst, with note $note

    python train.py \
     --train_path $tr \
     --test_path $tst \
     --demojize --lower_hashtag --segment_hashtag --textify_emoji\ # --add_cap_sign
     --mention_limit 0 --punc_limit 0 \
     --model mbert --time_pooling max_avg --layer 12 \
     --attention_probs_dropout_prob 0.1 --hidden_dropout_prob 0.3 \
     --lr 0.00002 --weight_decay 0.0 --layer_decrease 1.0 --freeze_upto -1 --warmup_ratio 0.1 \
     --batch_size 16 --train_step 700 --patience 20 --cuda 1 --note $note
done


