import numpy as np
import pandas as pd

ua_file_path = 'vocab/ukrainian_vocabulary.csv'

ua_df = pd.read_csv(ua_file_path, usecols=['Ukrainian Word', 'English Translation', 'Random Word 1', 'Random Word 2', 'Random Word 3'])

print(ua_df.iloc[3,0])

print(ua_df.head())
