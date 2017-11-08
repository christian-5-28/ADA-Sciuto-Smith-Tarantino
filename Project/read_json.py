import pandas as pd
import os
import glob
from matplotlib import pyplot as plt

'''
Loading all the data in one dictionary. You can access the json file from the dictionary
using the file name without the .json extension.
E.g.: all_data["condensed_2009"]
'''

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

# print(all_data["condensed_2009"])
# print(all_data["master_2009"])

sizes = [x.shape[0] for x in condensed]
years = [x.created_at[0].year for x in condensed]

plt.figure()
ax = plt.bar(sizes, years)
plt.show()
