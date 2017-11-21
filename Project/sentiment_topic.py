from Project.helpers import *
import pandas as pd

lexicons = load_lexicons()

print(lexicons.head())

'''Loading files'''
dir_path = 'topic_tweets'
directory = os.fsencode(dir_path)

first = True
tweets = pd.DataFrame()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if first:
        tweets = pd.read_json(dir_path + '/' + filename)
        first = False
    else:
        tweets.append(pd.read_json(dir_path + '/' + filename))



length = lexicons.shape[0]

# tweets = pd.read_json('topic_tweets/china.json')

tweets['anger'] = 0
tweets['anticipation'] = 0
tweets['disgust'] = 0
tweets['fear'] = 0
tweets['joy'] = 0
tweets['negative'] = 0
tweets['positive'] = 0
tweets['sadness'] = 0
tweets['surprise'] = 0
tweets['trust'] = 0

print('\nCHINA')

print(tweets.head())

for index, lexicon in lexicons.iterrows():
    print('Iterations to the end: %s...' % (length - index))
    # if index>100: # index = 100 => 31s
    #     break
    tweets_with_lexicon = tweets['text'].str.contains(lexicon.term, case=False)

    # We take the emotions
    emotions = lexicon.drop(['term'])

    # For every emotion we update the value in the df
    if any(tweets_with_lexicon.values): # faster
        for attribute, value in zip(emotions.index.values, emotions.values):
            if value != 0: # faster
                tweets.loc[tweets_with_lexicon, attribute] += value

print('\nCHINA AND EMOTIONS')

print(tweets.head())
