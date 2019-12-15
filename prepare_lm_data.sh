TRAIN_FILE='../resources/training.1600000.processed.noemoticon.csv'
CLEANED_FILE='../resources/tweet_corpus.txt'

cut -d ',' -f 6- $TRAIN_FILE | sed -e 's/^"//' -e 's/"$//' > $CLEANED_FILE



