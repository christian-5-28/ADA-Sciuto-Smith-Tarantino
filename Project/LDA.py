from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from Project.helpers import *
import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation
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
condensed_text = []
for x in condensed:
    temp = x[x.is_retweet == False]
    temp = temp[temp.source != "Twitter for iPhone"]
    condensed_text.append(temp.text.tolist())


flat_list = [item for sublist in condensed_text for item in sublist]

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
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stopwords)
tfidf = tfidf_vectorizer.fit_transform(flat_list)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# print(tfidf)
# tfidf (document, word-feature) tfij

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=no_features, stop_words=stopwords)
tf = tf_vectorizer.fit_transform(flat_list)
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

no_topics = 10

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
no_top_documents = 16
display_topics(lda_H, lda_W, tf_feature_names, flat_list, no_top_words, no_top_documents)

# display_topics(nmf_H, nmf_W, tfidf_feature_names, flat_list, no_top_words, no_top_documents)


# TODO: Find a way to make the same #makeamericagreatagain and MAKE AMERICA GREAT AGAIN.
# TODO: They finish in two different topics for NMF


