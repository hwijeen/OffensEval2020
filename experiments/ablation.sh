python ../train.py --emoji_min_freq 0 --mention_limit 0 --hashtag_min_freq 0 --punc_limit 0 --note none
python ../train.py --demojize --emoji_min_freq 0 --mention_limit 0 --hashtag_min_freq 0 --punc_limit 0 --note demojize
python ../train.py --emoji_min_freq 10 --mention_limit 0 --hashtag_min_freq 0  --punc_limit 0 --note min_emoji_10
python ../train.py --lower_hashtag --emoji_min_freq 0 --mention_limit 0 --hashtag_min_freq 0 --punc_limit 0 --note lower_hashtag
python ../train.py --emoji_min_freq 0 --mention_limit 0 --hashtag_min_freq 10 --punc_limit 0 --note none --note min_hashtag_10
python ../train.py --emoji_min_freq 0 --mention_limit 0 --hashtag_min_freq 0 --punc_limit 0 --add_cap_sign --note add_cap_sign
python ../train.py --emoji_min_freq 0 --mention_limit 3 --hashtag_min_freq 0 --punc_limit 0 --add_cap_sign --note mention_limit_3
python ../train.py --emoji_min_freq 0 --mention_limit 0 --hashtag_min_freq 0 --punc_limit 3 --add_cap_sign --note punc_limit_3
# replace_urls