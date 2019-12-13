for i in 0.2 0.5 0.8
do
    python train.py --mask_offensive $i --note mask_$i
done