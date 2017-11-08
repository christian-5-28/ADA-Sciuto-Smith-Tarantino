import pandas as pd
import numpy as np
import os
import glob
from Project.helpers import *
from matplotlib import pyplot as plt

all_data, condensed, master = load_data()

print(condensed[0].created_at.iloc[0].year)
sizes = [x.shape[0] for x in condensed]
print(sizes)
years = [x.created_at.iloc[0].year for x in condensed]
print(years)

plt.figure()
plt.bar(years, sizes)
plt.xticks(years)
plt.show()
