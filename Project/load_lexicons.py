import pandas as pd
import numpy as np

#TODO: quote https://medium.com/@aneesha/topic-modeling-with-scikit-learn-e80d33668730

# len 242060
lexicons = pd.read_table('lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Senselevel-v0.92.txt', sep='--|\t', names=('term', 'syn', 'emotion', 'value'), engine='python')

lexicons.syn = lexicons.syn.str.split(',')

if 'barbarism' in lexicons.term.values:
    print('yo')
else:
    print('nope')

print(lexicons.head())