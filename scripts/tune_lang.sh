#!/bin/bash
declare -A train_data
declare -A train_size
declare -A test_data

train_data[ar]=../data/2020/ar/offenseval-ar-training-v1-train.tsv
train_data[da]=../data/2020/da/offenseval-da-training-v1-train.tsv
train_data[el]=../data/2020/el/offenseval-greek-training-v1-train.tsv
train_data[en]=../data/2020/en/olid-training-v1.0-train.tsv
train_data[tr]=../data/2020/tr/offenseval-tr-training-v1-train.tsv

train_size[ar]=7201
train_size[da]=2665
train_size[el]=7870
train_size[en]=12690
train_size[tr]=28581

test_data[ar]=../data/2020/ar/offenseval-ar-training-v1-test.tsv
test_data[da]=../data/2020/da/offenseval-da-training-v1-test.tsv
test_data[el]=../data/2020/el/offenseval-greek-training-v1-test.tsv
test_data[en]=../data/2020/en/olid-training-v1.0-test.tsv
test_data[tr]=../data/2020/tr/offenseval-tr-training-v1-test.tsv

trn=${train_data[$1]}
trn_size=${train_size[$1]}
tst=${test_data[$1]}

layers=(11 12)
lrs=(0.00005 0.00001 0.0005)
warmup_ratio=(0 0.1)
batch_size=16
epoch=3
patience=1000

for i in {1..10}; do
    l=${layers[$(( $RANDOM % ${#layers[@]} ))]}
    lr=${lrs[$(( $RANDOM % ${#lrs[@]} ))]}
    w_r=${warmup_ratio[$(( $RANDOM % ${#warmup_ratio[@]} ))]}
    trn_step=$(( $trn_size / $batch_size * $epoch))

    python train.py \
     --train_path $trn --test_path $tst \
     --demojize --lower_hashtag --segment_hashtag --textify_emoji \
     --mention_limit 3 --punc_limit 3 \
     --model mbert --time_pooling max_avg --layer $l \
     --attention_probs_dropout_prob 0.1 --hidden_dropout_prob 0.3 \
     --lr $lr --weight_decay 0.0 --layer_decrease 1.0 --freeze_upto -1 --warmup_ratio $w_r \
     --batch_size $batch_size --train_step $trn_step --patience $patience --cuda 1 --note $1_tuning_$i
done
