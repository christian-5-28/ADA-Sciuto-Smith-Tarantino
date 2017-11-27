import json
import pandas as pd
import numpy as np

with open('tweet_topic_emotions/both_tweets_and_emotion_on_all.json', 'r') as fp:
    data = json.loads(fp.read())

tweets = pd.DataFrame(data)

tweets.created_at = tweets.created_at.apply(pd.to_datetime)

tweets_per_month = tweets.groupby(by=[tweets.topic, tweets.created_at.dt.year, tweets.created_at.dt.month]).count()[['anger']]
tweets_per_month = tweets_per_month.rename(columns={'anger': 'number_of_tweets'})

tweets_per_month.index.names = ['topic', 'year', 'month']
tweets_per_month.reset_index(level=['topic', 'year', 'month'], inplace=True)


tweets_per_month.year = tweets_per_month.year.astype(float)
tweets_per_month.month = tweets_per_month.month.astype(float)




# print(type(tweets_per_month.year.loc[0]))
#
saving_file = tweets_per_month.to_dict(orient='records')

with open('topic_tweets/topic_usage.json', 'w') as fp:
    json.dump(saving_file, fp)
