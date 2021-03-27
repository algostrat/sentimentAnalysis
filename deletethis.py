import pandas as pd

df = pd.read_csv('sent_scores1.csv')

print(df[df['abuse_prediction'].str.contains('yes')]['tweet'])
print(df)
print(type(df['abuse_prediction'][0]))
#print(df)
#df.loc[df['abuse_prediction'] == True]