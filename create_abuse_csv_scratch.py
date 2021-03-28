"""
This file is a demo to show every predicted abusive instance in the dataframe as
well as create the abuse_df.csv file that way you can view them in excel or notepad
"""

import pandas as pd

df = pd.read_csv('sent_scores1.csv')

abuse_df = df[df['abuse_prediction'].str.contains('yes')]['tweet']
abuse_df.to_csv('abuse_df.csv')
print(df[df['abuse_prediction'].str.contains('yes')]['tweet'])
#print(df)
#print(type(df['abuse_prediction'][0]))
#print(df)
#df.loc[df['abuse_prediction'] == True]