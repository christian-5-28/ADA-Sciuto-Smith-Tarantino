import pandas as pd
from Project.helpers import load_data
import numpy as np
from matplotlib import pyplot as plt


def select_time_interval(df, date_column, start_datetime, end_datetime):
    """
    returns a dataframe selected by a specific period of time
    """
    return df[(df[date_column] >= start_datetime) & (df[date_column] <= end_datetime)]


# retrieving all data from Trump's tweets dataset
all_data, condensed, master = load_data()

# getting the condensed version for year 2016 and 2017
condensed_2016 = all_data["condensed_2016"]
condensed_2017 = all_data["condensed_2017"]

# creating a dataframe for campaign period
cond_US_campaign_2016 = select_time_interval(condensed_2016, 'created_at',
                                             np.datetime64('2016-02-01'), np.datetime64('2016-11-08'))

cond_US_campaign_2016 = cond_US_campaign_2016.sort_values('created_at')

# creating a dataframe for president elect period
cond_pres_elect_df = select_time_interval(condensed_2016, 'created_at',
                                          np.datetime64('2016-11-09'), np.datetime64('2016-12-31'))

cond_pres_elect_df_2017 = select_time_interval(condensed_2017, 'created_at',
                                               np.datetime64('2017-01-01'), np.datetime64('2017-01-20'))

cond_pres_elect_df = cond_pres_elect_df.append(cond_pres_elect_df_2017)

cond_pres_elect_df = cond_pres_elect_df.sort_values('created_at')


# creating a dataframe for presidency period
cond_president_period_df = select_time_interval(condensed_2017, 'created_at',
                                                np.datetime64('2017-01-20'), np.datetime64('2017-11-05'))

cond_president_period_df = cond_president_period_df.sort_values('created_at')


# counting word usage of "fake", "news", fake news" in these 3 periods
cond_US_campaign_2016['fake_news_used'] = cond_US_campaign_2016['text'].str.contains('fake|news', case=False)
temp = cond_US_campaign_2016[cond_US_campaign_2016['fake_news_used'] == True]
cond_US_campaign_2016['Month'] = cond_US_campaign_2016['created_at'].dt.month
cond_US_campaign_2016['week/year'] = cond_US_campaign_2016['created_at'].apply(lambda x: "%d/%d" % (x.week, x.year))
temp = cond_US_campaign_2016[['Month', 'fake_news_used']]
temp = temp.groupby(['Month']).sum()
# print('count fake news usage campaign')
# print(temp)
# print(temp.fake_news_used.mean())

# counting word usage of "fake", "news", fake news" for president elect period
cond_pres_elect_df['fake_news_used'] = cond_pres_elect_df['text'].str.contains('fake|news', case=False)
'''
for text, match in zip(cond_pres_elect_df['text'], cond_pres_elect_df['fake_news_used']):
    if match:
        print(text)
'''

# print(cond_pres_elect_df['fake_news_used'].sum())
cond_pres_elect_df['Month'] = cond_pres_elect_df['created_at'].dt.month
cond_pres_elect_df['week/year'] = cond_pres_elect_df['created_at'].apply(lambda x: "%d/%d" % (x.week, x.year))
temp = cond_pres_elect_df[['Month', 'fake_news_used']]
temp = temp.groupby(['Month']).sum()
# print('count fake news usage pres elect')
# print(temp)
# print(temp.fake_news_used.mean())

# counting word usage of "fake", "news", fake news" presidency period:
cond_president_period_df['fake_news_used'] = cond_president_period_df['text'].str.contains('fake|news', case=False)

# adding month, date and week/year columns in order to make groupby operations
cond_president_period_df['Month'] = cond_president_period_df['created_at'].dt.month
cond_president_period_df['date'] = cond_president_period_df['created_at'].dt.date
cond_president_period_df['week/year'] = cond_president_period_df['created_at'].apply(lambda x: "%d/%d" % (x.week, x.year))
temp = cond_president_period_df[['week/year', 'fake_news_used']]
temp = temp.groupby(['week/year']).sum()

'''
print('count fake news usage pres period by week')
print(temp)
print(temp.fake_news_used.mean())
'''

# reorder by descending count of fake_news term usage by week
temp2 = temp.sort_values('fake_news_used', ascending=False)
'''
print('\n FAKE NEWS TERM USAGE BY WEEK ORDERED \n')
print(temp2)
print(temp2.shape)
'''
# adding the column of fake_news term usage by week to our dataframe
cond_president_period_df = cond_president_period_df.merge(temp2, left_on='week/year', right_index=True)


# fake news term usage by day:
temp = cond_president_period_df[['date', 'fake_news_used_x']]
temp = temp.groupby(['date']).sum()
temp = temp.sort_values('fake_news_used_x', ascending=False)

# adding the column of fake_news term usage by week to our dataframe
cond_president_period_df = cond_president_period_df.merge(temp, left_on='date', right_index=True)
cond_president_period_df = cond_president_period_df.rename(columns={'fake_news_used_y': 'fake_news_term_count_week',
                                                           'fake_news_used_x_y': 'fake_news_term_count_day'})


################################################################
################################################################

# loading fake news debunk files from WASHINGTON_POST
facts_checked = pd.read_json('fact_checked.json')

df_facts_checked = pd.DataFrame(facts_checked)

# merging our presidency dataframe with the facts_checked dataframe by the columns 'id_str' and 'tweet_id'
cond_president_period_df = cond_president_period_df.merge(df_facts_checked, left_on='id_str', right_on='tweet_id',
                                                          how='outer')

# saving our presidency period with fact checking in a csv file
# cond_president_period_df.to_csv('presidency_period_with_fact_check')

# temp cleaned contains only the tweets with analysis
temp_cleaned = cond_president_period_df.drop('in_reply_to_user_id_str', axis=1)
temp_cleaned = temp_cleaned.dropna()

# print(temp_cleaned)
# the merge was successful, we have 299 rows with analysis of fact checking


##################################
# checking the number of tweets from his android and iphone account
##################################

# we start with the campaign period
tweets_iphone_campaign_df = cond_US_campaign_2016.loc[cond_US_campaign_2016['source']
                                                      == 'Twitter for iPhone', ['Month', 'source']]

tweets_iphone_campaign_df.rename(columns={'source': 'tweets_from_iphone'}, inplace=True)

tweets_android_campaign_df = cond_US_campaign_2016.loc[cond_US_campaign_2016['source']
                                                       == 'Twitter for Android', ['Month', 'source']]

tweets_android_campaign_df.rename(columns={'source': 'tweets_from_android'}, inplace=True)

tweets_source_campaign_df = tweets_iphone_campaign_df.append(tweets_android_campaign_df)
# print(tweets_source_campaign_df)
# print(tweets_source_campaign_df)

tweets_source_campaign_month = tweets_source_campaign_df.groupby('Month').agg({'tweets_from_iphone': 'count',
                                                                               'tweets_from_android': 'count'})

tweets_source_campaign_month.sort_index(inplace=True)

# creating a bar chart to visualize our results
barchart_tweets_source = tweets_source_campaign_month.plot(kind='bar', title="tweets source campaign",
                                                           legend=True, fontsize=12, figsize=(20, 10))

barchart_tweets_source.set_xlabel("Month", fontsize=12)
barchart_tweets_source.set_ylabel("number of tweets", fontsize=12)


# Now we do the same process for the presidency period

tweets_iphone_presidency_df = cond_president_period_df.loc[cond_president_period_df['source']
                                                           == 'Twitter for iPhone', ['Month', 'source']]

tweets_iphone_presidency_df.rename(columns={'source': 'tweets_from_iphone'}, inplace=True)

tweets_android_presidency_df = cond_president_period_df.loc[cond_president_period_df['source']
                                                           == 'Twitter for Android', ['Month', 'source']]

tweets_android_presidency_df.rename(columns={'source': 'tweets_from_android'}, inplace=True)

tweets_source_presidency_df = tweets_iphone_presidency_df.append(tweets_android_presidency_df)


# now we groupby month and we count the number of tweets for the two columns
tweets_source_presidency_month = tweets_source_presidency_df.groupby('Month').agg({'tweets_from_iphone': 'count',
                                                                                   'tweets_from_android': 'count'})

tweets_source_presidency_month.sort_index(inplace=True)

# creating a bar chart to visualize our results
barchart_tweets_source_pres = tweets_source_presidency_month.plot(kind='bar', title="tweets source presidency",
                                                                legend=True, fontsize=12, figsize=(20, 10))

barchart_tweets_source_pres.set_xlabel("Month", fontsize=12)
barchart_tweets_source_pres.set_ylabel("number of tweets", fontsize=12)

# showing the two plots
# plt.show()

# as result we obtain that from march 2017 the president changed is phone with an iphone.


##################################
# test dataframe for retweets
##################################

test_df_retweets = cond_president_period_df[cond_president_period_df['is_retweet']]

print(test_df_retweets.shape)

##################################
# test dataframe for his replies
##################################

mask = pd.notnull(cond_president_period_df['in_reply_to_user_id_str'])

test_df_replies = cond_president_period_df[mask]

print(test_df_replies.shape)

