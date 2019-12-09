for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
    python train.py --pooling avg --demojize --lower_hashtag --cuda 1 --data_size $i
done
