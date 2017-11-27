from Project.helpers import *
import pandas as pd
import json

# lexicons = load_lexicons()
#
# # print(lexicons.head())
#
# '''Loading files'''
# dir_topic_path = 'topic_tweets'
# directory_topic = os.fsencode(dir_topic_path)
#
# dir_total_path = 'trump_tweets'
# directory_total = os.fsencode(dir_total_path)
#
# json_files = []
# ids = []
#
# first = True
# tweets = pd.DataFrame()
# for file in os.listdir(directory_topic):
#     filename = os.fsdecode(file)
#
#     if first:
#         tweets = pd.read_json(dir_topic_path + '/' + filename)
#         first = False
#
#     else:
#         tweets = tweets.append(pd.read_json(dir_topic_path + '/' + filename))
#
#     ids += tweets['id'].tolist()
#
# ids = set(ids)
#
# for file in os.listdir(directory_total):
#     filename = os.fsdecode(file)
#     if 'condensed' in filename:
#         total_tweets = pd.read_json(dir_total_path + '/' + filename)
#         total_tweets = total_tweets[~total_tweets.id_str.isin(ids)]
#         total_tweets = total_tweets.rename(columns={'id_str': 'id'})
#         tweets = tweets.append(total_tweets)
#
# tweets['anger'] = 0
# tweets['anticipation'] = 0
# tweets['disgust'] = 0
# tweets['fear'] = 0
# tweets['joy'] = 0
# tweets['negative'] = 0
# tweets['positive'] = 0
# tweets['sadness'] = 0
# tweets['surprise'] = 0
# tweets['trust'] = 0
#
# length = lexicons.shape[0]
#
# for index, lexicon in lexicons.iterrows():
#     print('Iterations to the end: %s...' % (length - index))
#     # if index>100: # index = 100 => 31s
#     #     break
#     tweets_with_lexicon = tweets['text'].str.contains(lexicon.term, case=False)
#
#     # We take the emotions
#     emotions = lexicon.drop(['term'])
#
#     # For every emotion we update the value in the df
#     if any(tweets_with_lexicon.values):  # faster
#         for attribute, value in zip(emotions.index.values, emotions.values):
#             if value != 0:  # faster
#                 tweets.loc[tweets_with_lexicon, attribute] += value
#
#
# tweets.created_at = tweets.created_at.dt.strftime("%Y-%m-%d %H:%M:%S")
# saving_file = tweets.to_dict(orient='records')
#
# with open('tweet_topic_emotions/both_tweets_and_emotion_on_all.json', 'w') as fp:
#     json.dump(saving_file, fp)

with open('tweet_topic_emotions/both_tweets_and_emotion_on_all.json', 'r') as fp:
    data = json.loads(fp.read())

