python train.py --demojize --lower_hashtag --patience 100 --cuda 0 --note mask_none_baseline
for MAX in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9;
do
  python train.py --demojize --lower_hashtag --patience 100 --cuda 1 --mask_offensive $MAX --note mask_offensive_const_${MAX}
done