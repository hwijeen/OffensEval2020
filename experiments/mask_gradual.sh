for i in 0.001 0.01 0.05 0.1;
do for MAX in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0;
do
  python train.py --demojize --lower_hashtag --patience 100 --cuda 0 --mask_offensive $MAX --mask_gradual $i --mask_increase --note mask_offensive_max_${MAX}_grad_${i}_increase;
done
done
