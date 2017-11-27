from Project.helpers import *
import pandas as pd
import collections
import json
import re
import operator


def clean_sentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

'''Loading files'''
dir_topic_path = 'topic_tweets'
directory_topic = os.fsencode(dir_topic_path)

json_files = []

first = True
tweets = pd.DataFrame()
for file in os.listdir(directory_topic):
    filename = os.fsdecode(file)

    if filename == '.DS_Store':
        continue

    if first:
        print(dir_topic_path + '/' + filename)
        tweets = pd.read_json(dir_topic_path + '/' + filename)
        first = False

    else:
        tweets = tweets.append(pd.read_json(dir_topic_path + '/' + filename))

print(tweets.head())

topics = tweets.topic.unique()

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
stopwords = ['the', 'to', 'is', 'a', 'and', 'in', 'you', 'of', 'i', 'for', 'at', 'on',
             'be', 'amp', 'your', 'my', 'it', 'will', 'our', 'us', 'we', 'cont', 'are',
             'from', 'has', 'that', 'this', 'she', 'her', 'have', 'with', 'he',
             'new', 'just', 'from', 'now', 'as', 'he', 'its', 'by', 'they', 'was',
             'not', 'so', 'more', 'about', 'what', 'all', 'get', 'but', 'one',
             'over', 'their', 'why', 'when', 'what', 'them', 'who', 'said', 'out',
             'would', 'had', 'can', 'should', 'would', 'do', 'been', 'an']

for topic in topics:
    topic_dict = {}
    temp = tweets[tweets.topic == topic]

    for index, row in temp.iterrows():
        text = clean_sentences(row.text.lower()).split()

        text = [word for word in text if word not in stopwords]

        counter = collections.Counter(text)
        for word, count in counter.items():

            if word in topic_dict:
                topic_dict[word] += count
            else:
                topic_dict[word] = count
    print(topic)
    print(dict(sorted(topic_dict.items(), key=operator.itemgetter(1), reverse=True)[:20]))
    ordered_dict = dict(sorted(topic_dict.items(), key=operator.itemgetter(1), reverse=True))
    # saving_file = []
    # for key, value in ordered_dict.items():
    #     saving_file.append([key, value])
    # with open('data/'+topic+'_word_usage.json', 'w') as fp:
    #     json.dump(saving_file, fp)


