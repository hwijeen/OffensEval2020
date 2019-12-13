TEST_FILE=../data/olid-test-v1.0.tsv
if [ -f "$TEST_FILE" ]; then
  echo "$TEST_FILE exists"
else
  python merge_test.py
fi
#python train.py --demojize --emoji_min_freq 5 --mention_limit 0 --hashtag_min_freq 0 --punc_limit 0 --note demojize_min_5
#python train.py --demojize --emoji_min_freq 5 --mention_limit 3 --hashtag_min_freq 0 --punc_limit 0 --note demojize_min_5_mention_3
#python train.py --emoji_min_freq 0 --lower_hashtag --mention_limit 3 --hashtag_min_freq 0 --punc_limit 0 --note lower_hashtag_mention_3
#python train.py --emoji_min_freq 0 --mention_limit 0 --lower_hashtag --hashtag_min_freq 5 --punc_limit 0 --note lower_hashtag_min_5
python train.py --demojize --emoji_min_freq 10 --mention_limit 3 --lower_hashtag --hashtag_min_freq 10 --punc_limit 3 --note all
python train.py --emoji_min_freq 10 --mention_limit 3 --lower_hashtag --hashtag_min_freq 10 --punc_limit 3 --note ab_demojize
python train.py --demojize --emoji_min_freq 0 --mention_limit 3 --lower_hashtag --hashtag_min_freq 10 --punc_limit 3 --note ab_emoji_min_freq
python train.py --demojize --emoji_min_freq 10 --mention_limit 0 --lower_hashtag --hashtag_min_freq 10 --punc_limit 3 --note ab_mention_limit
python train.py --demojize --emoji_min_freq 10 --mention_limit 3 --hashtag_min_freq 10 --punc_limit 3 --note ab_lower_hashtag
python train.py --demojize --emoji_min_freq 10 --mention_limit 3 --lower_hashtag --hashtag_min_freq 0 --punc_limit 3 --note ab_hashtag_min_freq
python train.py --demojize --emoji_min_freq 10 --mention_limit 3 --lower_hashtag --hashtag_min_freq 10 --punc_limit 0 --note ab_punc_limit