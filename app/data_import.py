import pandas as pd
import sqlite3

#Create a database
conn = sqlite3.connect('../data/db.sqlite')
conn.close()

# Connect to the database
conn = sqlite3.connect('../data/db.sqlite')

# Import CSV into database
df = pd.read_csv('../data/ratings.csv')

df.to_sql('ratings', conn, index=False, if_exists='replace', dtype={'id': 'INTEGER PRIMARY KEY'})

conn.close()