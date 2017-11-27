import requests
from bs4 import BeautifulSoup

#TODO: quote https://www.ranks.nl/stopwords

URL_sw = 'https://www.ranks.nl/stopwords'

r = requests.get(URL_sw, verify=False)

soup = BeautifulSoup(r.text, 'html.parser')

div = soup.find(lambda tag: tag.name=='div' and tag.has_attr('id') and tag['id']=="article09e9cfa21e73da42e8e88ea97bc0a432")
table = div.find(lambda tag: tag.name=='table')

stopwords = []

cols = table.findAll('td')
for col in cols:
    col = str(col)
    col = col.replace('<td style="width: 33%;" valign="top">', '')
    col = col.replace('<td valign="top">', '')
    words = col.split("<br/>")
    stopwords.extend(words)

import json
with open('stopwords_2.json', 'w') as fp:
    json.dump(stopwords, fp)

