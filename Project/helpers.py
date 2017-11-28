import pandas as pd
import os
import glob
import json
import requests
from bs4 import BeautifulSoup


def load_lexicons():
    """
    Loading the 'emotions' and return a df with a unique column a word and 10 other columns of emotions
    referred to that term
    """

    lexicons = pd.read_table('lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt', sep='\t',
                             names=('term', 'emotion', 'value'))

    terms = lexicons.term

    emotions = lexicons.drop('term', axis=1).pivot(columns='emotion', values='value')
    emotions['term'] = terms
    emotions.fillna(0, inplace=True)
    emotions = emotions.groupby('term').sum()
    emotions.reset_index(inplace=True)

    return emotions


def load_data():
    """
    Loading all the data in one dictionary and two lists: condensed and master and returning them.
    You can access the json file from the dictionary
    using the file name without the .json extension.
    E.g.: all_data["condensed_2009"]
    """

    trump_tweets = glob.glob("trump_tweets/*.json")

    all_data = {}
    condensed = []
    master = []

    for json_file in trump_tweets:

        file = pd.read_json(json_file)
        all_data[os.path.basename(json_file).replace(".json", "")] = file

        if "master" in os.path.basename(json_file):
            master.append(file)
        else:
            condensed.append(file)

    return all_data, condensed, master


def get_stopwords_from_html():
    """
    Saving stopwords from url
    """

    URL_sw = 'https://www.ranks.nl/stopwords'

    r = requests.get(URL_sw, verify=False)

    soup = BeautifulSoup(r.text, 'html.parser')

    div = soup.find(
        lambda tag: tag.name == 'div' and tag.has_attr('id') and tag['id'] == "article09e9cfa21e73da42e8e88ea97bc0a432")
    table = div.find(lambda tag: tag.name == 'table')

    stopwords = []

    cols = table.findAll('td')
    for col in cols:
        col = str(col)
        col = col.replace('<td style="width: 33%;" valign="top">', '')
        col = col.replace('<td valign="top">', '')
        words = col.split("<br/>")
        stopwords.extend(words)

    with open('stopwords/stopwords_2.json', 'w') as fp:
        json.dump(stopwords, fp)


def select_time_interval(df, date_column, start_datetime, end_datetime):
    """
    returns a dataframe selected by a specific period of time
    """
    return df[(df[date_column] >= start_datetime) & (df[date_column] <= end_datetime)]


def get_hillary_tweets_16_17(all_data):
    '''Retrieving tweets concernig hillary from 2016 and 2017'''

    # TOTAL
    # TOTAL HILLARY = 962
    # FROM 2016 - 2017 = 698

    # NO IPHONE
    # TOTAL HILLARY = 669
    # FROM 2016 - 2017 = 415

    # FROM POLITICS = 15!!!

    condensed_2016 = all_data['condensed_2016']
    condensed_2017 = all_data['condensed_2017']
    condensed = []
    condensed.append(condensed_2016)
    condensed.append(condensed_2017)

    condensed_text = []
    condensed_date = []
    condensed_retweet_count = []
    condensed_favorite_count = []
    condensed_id = []
    for x in condensed:
        temp = x[x.is_retweet == False]
        temp = temp[temp.source != "Twitter for iPhone"]
        condensed_text.append(temp.text.tolist())
        condensed_date.append(temp.created_at.tolist())
        condensed_retweet_count.append(temp.retweet_count.tolist())
        condensed_favorite_count.append(temp.favorite_count.tolist())
        condensed_id.append(temp.id_str.tolist())

    flat_list = [item for sublist in condensed_text for item in sublist]
    flat_list_date = [item for sublist in condensed_date for item in sublist]
    flat_list_retweet_count = [item for sublist in condensed_retweet_count for item in sublist]
    flat_list_favorite_count = [item for sublist in condensed_favorite_count for item in sublist]
    flat_list_id = [item for sublist in condensed_id for item in sublist]

    tweets = []
    flat_list_info = []
    for tweet, date, ret, fav, id_ in zip(flat_list, flat_list_date, flat_list_retweet_count, flat_list_favorite_count, flat_list_id):
        if ('hillary' in tweet) or ('clinton' in tweet) or ('Hillary' in tweet) or ('Clinton' in tweet) \
                or ('Crooked' in tweet) or ('crooked' in tweet):

            tweets.append(tweet)
            flat_list_info.append((date, ret, fav, id_))

    return tweets, flat_list_info


def topic_to_json(topic, json_list, flat_list_info, topic_str):
    for tweet in topic:
        date, ret, fav, id_ = flat_list_info[tweet[1]]

        json = {
            'created_at': str(date),
            'id': id_,
            'retweet_count': ret,
            'favorite_count': fav,
            'text': tweet[0],
            'topic': topic_str
        }

        json_list.append(json)


def topic_to_json_hillary(hillary_topic, hillary_json, hillary_info, internal=False):
    for index, tweet in enumerate(hillary_topic):
        date, ret, fav, id_ = hillary_info[index]

        if not internal:
            json = {
                'created_at': str(date),
                'id': id_,
                'retweet_count': ret,
                'favorite_count': fav,
                'text': tweet,
                'topic': 'hillary'
            }

        else:
            json = {
                'created_at': str(date),
                'id': id_,
                'retweet_count': ret,
                'favorite_count': fav,
                'text': tweet,
                'topic': 'internal_politics'
            }

        hillary_json.append(json)

