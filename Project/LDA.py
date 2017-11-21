from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from Project.helpers import *
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
import json
import collections
from nltk import PorterStemmer
from nltk import word_tokenize

#TODO: quote http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm


with open('stopwords_joined.json', 'r') as f:
         stopwords = json.load(f)

all_data, condensed, master = load_data()

'''
From all the condensed data, here it is created a list of strings. Every element of the list
contains a string that contains all the tweets of one year, separated by /n
'''
# condensed_text = [x.text.tolist() for x in condensed]
# condensed_text = ["\n".join(text) for text in condensed_text]

# condensed_text = ["\n".join(x.text.tolist()) for x in condensed]

'''
From all the condensed data, here it is created a list of strings. Every element of the list
is a tweet
'''

'''
That are not a retweet - and are not from his staff -> iPhone: http://varianceexplained.org/r/trump-tweets/
'''
# TODO: quote http://varianceexplained.org/r/trump-tweets/
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


flat_list_text = [item for sublist in condensed_text for item in sublist]
flat_list_date = [item for sublist in condensed_date for item in sublist]
flat_list_retweet_count = [item for sublist in condensed_retweet_count for item in sublist]
flat_list_favorite_count = [item for sublist in condensed_favorite_count for item in sublist]
flat_list_id = [item for sublist in condensed_id for item in sublist]

# Date, retweet_count, favorite_count
flat_list_info = []
for date, ret, fav, id_ in zip(flat_list_date, flat_list_retweet_count, flat_list_favorite_count, flat_list_id):
    flat_list_info.append((date, ret, fav, id_))

'''
STEMMING

In order to use word_tokenize you have to type:
import nltk
nltk.download()
go to model and then install punkt
'''
# ps = PorterStemmer()
#
# new_flat_list = []
#
# for tweet in flat_list:
#     a = []
#     # print(tweet + '\n\n')
#     for w in word_tokenize(tweet):
#         # print(w)
#         a.append(ps.stem(w))
#
#     new_flat_list.append(" ".join(a))

# print(len(new_flat_list))
# print(len(flat_list))
# 31942

'''
TFij = Nij/Dj
IDF = log10 (D/number of documents that contain i)
The two terms are multiplied in order to obtain the importance of the word in the document
'''

'''
A tf-idf transformer is applied to the bag of words matrix that NMF must process with the TfidfVectorizer. 
LDA on the other hand, being a probabilistic graphical model (i.e. dealing with probabilities) only requires
raw counts, so a CountVectorizer is used. Stop words are removed and the number of terms included in the bag of
words matrix is restricted to the top 1000.
'''

no_features = 1000

# NMF is able to use tf-idf (tf*idf)
# tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stopwords)
# tfidf = tfidf_vectorizer.fit_transform(flat_list)
# tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# print(tfidf)
# tfidf (document, word-feature) tfij

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stopwords)
tf = tf_vectorizer.fit_transform(flat_list_text)
tf_feature_names = tf_vectorizer.get_feature_names()

'''
As mentioned previously the algorithms are not able to automatically determine the number 
of topics and this value must be set when running the algorithm. Comprehensive documentation on
available parameters is available for both NMF and LDA. Initialising the W and H matrices in NMF 
with ‘nndsvd’ rather than random initialisation improves the time it takes for NMF to converge.
LDA can also be set to run in either batch or online mode.

We need to obtain the word to topics matrix (H) and the topics to documents matrix (W) from both the NMF and LDA 
algorithms. The word to topics matrix (H) can be obtained from the component_ attribute of the model after .fit() 
is called. Getting the topics to document matrix, is a little tricky but after reading the Scikit Learn api documents 
for each algorithm it will all make sense. Calling the transform() method on the algorithm model will return the topic 
to document matrix (W).
'''

no_topics = 8

# Run NMF
# nmf_model = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
# nmf_W = nmf_model.transform(tfidf)
# nmf_H = nmf_model.components_

# Run LDA
lda_model = LatentDirichletAllocation(n_components=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)
lda_W = lda_model.transform(tf)
lda_H = lda_model.components_

'''
H: every rows is a topic, every column is a word (columns length = no_features)
Every topic contains the weight of every word about that topic.
For this reason, if we do an inverse argsort of a topic (of a row), that returns the indices of the ordered list
and we use those indices to extract words from feature_names, these words will be ordered from the most
to the least important.

W: every rows is a document, every column is a topic. The same concept presented above for documents and topic.
'''
# print(nmf_H.shape) = (7, 1000)
# print(nmf_H)
# print(nmf_W.shape) = (31942, 7)
# print(nmf_W)

'''
The structure of the resulting matrices returned by both NMF and LDA is the same and the Scikit Learn interface
to access the returned matrices is also the same. This is great and allows for a common Python method that is able 
to display the top words in a topic. Topics are not labeled by the algorithm — a numeric index is assigned.

It takes both the words to topics matrix (H) and the topics to documents matrix (W) as arguments. The method also needs 
to take the document collection (documents) and number of top documents (no_top_documents) to display in additional 
to the words (feature_names) and number of top words (no_top_words) to display as arguments. The display_topics method 
prints out a numerical index as the topic name, prints the top words in the topic and then prints the top documents in 
the topic. The top words and top documents have the highest weights in the returned matrices. The argsort() method is 
used to sort the row or column of the matrix and returns the indexes for the cells that have the highest weights in order.
'''


def display_topics(H, W, feature_names, documents, no_top_words, no_top_documents):
    for topic_idx, topic in enumerate(H):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        top_doc_indices = np.argsort(W[:, topic_idx])[::-1][0:no_top_documents]
        for doc_index in top_doc_indices:
            print(documents[doc_index])
            print('Score: ' + str(W[doc_index, topic_idx]))
            print('\n')


no_top_words = 8
no_top_documents = 20
# display_topics(lda_H, lda_W, tf_feature_names, flat_list, no_top_words, no_top_documents)

# display_topics(nmf_H, nmf_W, tfidf_feature_names, flat_list, no_top_words, no_top_documents)

'''
Topics:
0: POLITICS
1: HOTELS AND GOLF
2: POLITICS
3: CELEBRITY APPRENTICE AND SHOWS
4: BOOKS, YANKEES, VARIOUS
5: GOLF, TRUMP'S BUSINESSES
6: QUOTATIONS, VARIOUS
7: INTERVIEWS, DEBATES

First I'd put together topics 1-5 in TRUMP BUSINESS CATEGORY
Second I'd put together 4 - 6 in VARIOUS CATEGORY

Third, the idea is to put together the best documents from 0 and 2 (score > 75%)
and divide this topic in other topics.

'''

'''
Putting together 0-2:
This new topic will contain EVERYTHING THAT REFERS TO POLITICS.

Putting together 1-5:
This new topic will contain GOLF, HOTELS, COLOGNE, Trump Signature Collection.

Putting together 4-6:
This new topic will contain GENERAL TWEETS, BOOKS, SOME QUOTATIONS, ... .

'''
business = []
scores_business = []
various = []
scores_various = []
politics = []
scores_politics = []
shows = []
scores_shows = []
interviews = []
scores_interviews = []

'''In every topic list we append the text and the index of the tweet in order to get the info later'''

for topic_idx, topic in enumerate(lda_H):

    # Taking the indices of the documents ordered by importance in the topic
    top_doc_indices_ordered = np.argsort(lda_W[:, topic_idx])[::-1]

    for doc_index in top_doc_indices_ordered:
        score = lda_W[doc_index, topic_idx]

        # If the documents are less important than 0.60, I finish collecting documents for that topic
        if (score < 0.60):
            break

        # BUSINESS - score no less than 0.6
        if topic_idx == 1 or topic_idx == 5:

            business.append([flat_list_text[doc_index], doc_index])
            scores_business.append(score)

        # VARIOUS - score can be 0.5
        elif topic_idx == 4 or topic_idx == 6:

            various.append([flat_list_text[doc_index], doc_index])
            scores_various.append(score)

        # POLITICS - score no less than 0.6
        elif topic_idx == 0 or topic_idx == 2:
            politics.append([flat_list_text[doc_index], doc_index])
            scores_politics.append(score)

        # CELEBRITY APPRENTICE AND SHOWS - score no less than 0.6
        elif topic_idx == 3:

            shows.append([flat_list_text[doc_index], doc_index])
            scores_shows.append(score)

        # INTERVIEWS AND DEBATES - score no less than 0.6
        elif topic_idx == 7:

            interviews.append([flat_list_text[doc_index], doc_index])
            scores_interviews.append(score)



'''PRINT BUSINESS TOPIC'''
# scores_business_idx_ordered = np.argsort(scores_business)[::-1]
# print("NEW TOPIC: BUSINESS")
# for index in scores_business_idx_ordered:
#     print(business[index])
#     print(scores_business[index])
#     print()

'''PRINT VARIOUS TOPIC'''
# scores_various_idx_ordered = np.argsort(scores_various)[::-1]
# print("NEW TOPIC: VARIOUS ")
# for index in scores_various_idx_ordered:
#     print(various[index])
#     print(scores_various[index])
#     print()

'''PRINT POLITICS TOPIC'''
# scores_politcs_idx_ordered = np.argsort(scores_politics)[::-1]
# print("NEW TOPIC: VARIOUS ")
# for index in scores_politcs_idx_ordered:
#     print(politics[index])
#     print(scores_politics[index])
#     print()

# Arrivato a controllare qua : 0.769558751189

'''PRINT SHOWS TOPIC'''
# scores_shows_idx_ordered = np.argsort(scores_shows)[::-1]
# print("NEW TOPIC: SHOWS ")
# for index in scores_shows_idx_ordered:
#     print(shows[index])
#     print(scores_shows[index])
#     print()

'''PRINT INTERVIEWS AND DEBATES TOPIC'''
# scores_interviews_idx_ordered = np.argsort(scores_interviews)[::-1]
# print("NEW TOPIC: INTERVIEWS AND DEBATES ")
# for index in scores_interviews_idx_ordered:
#     print(interviews[index])
#     print(scores_interviews[index])
#     print()


# print(len(politics))
# print(len(business))
# print(len(various))
# print(len(shows))
# print(len(interviews))


'''
In Politics Topic hillary appears only 15 times, probably the topic detection didn't count her.
For this reason we will use a function to retrieve her tweets
'''

hillary_topic, hillary_info = get_hillary_tweets_16_17(all_data)

# Removing the tweets in politics that contain hillary, because we are going to add every tweet that refers to hillary
# later and we don't want to have duplicates
for tweet in list(politics):
    if ('hillary' in tweet[0]) or ('clinton' in tweet[0]) or ('Hillary' in tweet[0]) or ('Clinton' in tweet[0]):
        politics.remove(tweet)


'''
Searching for keywords inside the politics tweets in order to get more specific topics
'''

china_keywords = ['china', "China", 'Chinese']
obama_keywords = ['Obama', 'obama', '@BarackObama', 'Obamacare', 'obamacare', '#Obamacare', 'OBAMACARE']
hillary_keywords = ['hillary', 'clinton', 'Hillary', 'Clinton', 'Crooked', 'crooked']
foreign_politics_keywords = ['iran', 'Iran', 'Iraq', 'iraq', 'oil', 'Oil', 'nuclear', 'gas', 'Lybia', 'ebola',
                             'Ebola', 'Japan', 'foreign', 'trade', 'Saudi Arabia', '#IranDeal', 'Greece',
                             'Middle East', 'OPEC', 'terrorists', 'terrorist',  'ISIS', 'isis', '#isis', '#ebola',
                             'Afghanistan', 'Russia', 'russia', 'Snowden', 'nations', 'Georgia',
                             'military spending', 'Pakistan', 'Bin Laden', 'N.A.T.O', 'Assad', 'Libya',
                             'Solders', 'Soldiers', 'soldiers' 'weapon', 'weapons', 'Al Qaeda', 'Putin', 'military',
                             'Mexico', 'Europe', 'South Korea', 'military', 'Canada', 'Mubarak', 'Islamists',
                             'TERRORISTS', 'OIL', 'Afghan'] + china_keywords

internal_politics_keywords = ['Government', 'government', 'govt.', 'Washington', 'republicans', 'Republicans', 'Repubs',
                              'Democrats', 'democrats', 'tax', 'taxes', 'debt', 'jobs', 'job', 'borders', 'border',
                              'Wall Street', 'deficit', 'immigrants', 'employment', 'unemployment', 'deficits',
                              'deficit', 'taxing', 'Senate', 'Senator', 'senate', 'senator', 'our country',
                              'Congress', '@whitehouse', 'Taxpayer', 'Our whole country', 'officials',
                              'Medicare', 'minority', 'death penalty', 'infrastructure', 'House', 'Fiscal',
                              'military spending', 'Social Security', 'Medicare', 'Weak leaders',
                              'Cruz', 'leadership', 'citizen', 'economic', 'our great country', 'political', 'Pentagon',
                              'candidate', 'Fed', 'Amendment', '#TedCruz', 'defense cuts', 'drug abuse',
                              'food stamps', 'Jimmy Carter', 'guns are outlawed', 'candidate', 'Parties', 'Reagan',
                              'Chafee', 'Insurance', 'civil liberties', 'Federal Court', 'Supreme Court', 'CIA',
                              'lobbyists'] + obama_keywords #  + hillary

trump_keywords = ['@realDonaldTrump', 'Trump', 'trump', 'Donald', 'Jeb', 'Bush', '#Trump',
                  'MAKE AMERICA GREAT AGAIN', 'Make America Great Again', '#TimeToGetTough', 'Trumpative',
                  '#DonaldTrump', 'TRUMP', 'I will', 'VICTORY', 'no other leader', 'DONALD', 'DONALD TRUMP',
                  'TRUMP 2016', 'Making America Great Again', 'H-1B reform', 'MR.TRUMP For President', 'total pro',
                  'rallies', 'run for president', 'running for President', '#MakeAmericaGreatAgain',
                  'Big speech', 'great wall']



china_topic = []
obama_topic = []
foreign_politics_topic = []
internal_politics_topic = []
trump_4pre_topic = []
not_got = []

china_json = []
obama_json = []
foreign_politics_json = []
internal_politics_json = []
trump_4pre_json = []

for tweet in politics:

    control = 0

    date, ret, fav, id_ = flat_list_info[tweet[1]]

    json_file = {
        'created_at': str(date),
        'id': id_,
        'retweet_count': ret,
        'favorite_count': fav,
        'text': tweet[0],
        'topic': ''
    }

    for keyword in china_keywords:
        if keyword in tweet[0]:
            china_topic.append(tweet)

            json_file['topic'] = 'china'
            china_json.append(json_file.copy())

            control += 1
            break

    for keyword in obama_keywords:
        if keyword in tweet[0]:
            obama_topic.append(tweet)

            json_file['topic'] = 'obama'
            obama_json.append(json_file.copy())

            control += 1
            break

    for keyword in foreign_politics_keywords:
        if keyword in tweet[0]:
            foreign_politics_topic.append(tweet)

            json_file['topic'] = 'foreign_politics'
            foreign_politics_json.append(json_file.copy())
            # foreign_politics_json.append(copy.deepcopy(json))

            control += 1
            break

    for keyword in internal_politics_keywords:
        if keyword in tweet[0]:
            internal_politics_topic.append(tweet)

            json_file['topic'] = 'internal_politics'
            internal_politics_json.append(json_file.copy())

            control += 1
            break

    for keyword in trump_keywords:
        if keyword in tweet[0]:
            trump_4pre_topic.append(tweet)

            json_file['topic'] = 'trump_politcs'
            trump_4pre_json.append(json_file.copy())

            control += 1
            break

    if control == 0:
        not_got.append(tweet)





'''
TO JSON
'''
business_json = []
various_json = []
shows_json = []
interviews_json = []
hillary_json = []


topic_to_json(business, business_json, flat_list_info, 'business')
topic_to_json(various, various_json, flat_list_info, 'various')
topic_to_json(shows, shows_json, flat_list_info, 'shows')
topic_to_json(interviews, interviews_json, flat_list_info, 'interviews_debates')
topic_to_json_hillary(hillary_topic, hillary_json, hillary_info)
topic_to_json_hillary(hillary_topic, internal_politics_json, hillary_info, internal=True) # internal + hillary

with open('topic_tweets/business.json', 'w') as fp:
    json.dump(business_json, fp)

with open('topic_tweets/various.json', 'w') as fp:
    json.dump(various_json, fp)

with open('topic_tweets/shows.json', 'w') as fp:
    json.dump(shows_json, fp)

with open('topic_tweets/interviews.json', 'w') as fp:
    json.dump(interviews_json, fp)

with open('topic_tweets/hillary.json', 'w') as fp:
    json.dump(hillary_json, fp)

with open('topic_tweets/china.json', 'w') as fp:
    json.dump(china_json, fp)

with open('topic_tweets/obama.json', 'w') as fp:
    json.dump(obama_json, fp)

with open('topic_tweets/foreign_politics.json', 'w') as fp:
    json.dump(foreign_politics_json, fp)

with open('topic_tweets/internal_politics.json', 'w') as fp:
    json.dump(internal_politics_json, fp)





'''
That's how we we will count the most common word in a topic
'''
#TODO: add a second stopword file for punctuation
# words_in_topic = []
# for tweet in china_topic:
#     words = word_tokenize(tweet)
#     for word in list(words):
#         if word in stopwords:
#             words.remove(word)
#     words_in_topic += words
#
# counter = collections.Counter(words_in_topic)
# print(counter.most_common())

