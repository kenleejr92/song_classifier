"""
Scrape the lyrics for the datamining project
	- Given an artist name and song name, scrape the lyrics

@author - Alan

"""

import pymysql.cursors
import pandas as pd
import numpy as np

# read csv file containing id, song, artist
data = pd.read_csv('./info.csv', delimiter = ",", names=['ID','Song','Artist']);
data = data.replace(np.nan, '', regex=True)

# establish connection to sql server
connection = pymysql.connect(host='127.0.0.1',\
   user='root',password='root',db='songs')

cursor = connection.cursor()

# create sql table (only need to do this once)
sql = '''CREATE TABLE songs (
songID VARCHAR(50) PRIMARY KEY, 
artist VARCHAR(50) DEFAULT NULL,
title VARCHAR(100) DEFAULT NULL,
INDEX artist (artist)
);'''
cursor.execute(sql)
connection.commit()

# populate sql table with data
for index, row in data.iterrows():
    
    query = "INSERT INTO songs (songID, artist, title) VALUES ('%s', '%s', '%s')" % (row['ID'], row['Artist'].replace("'", ""),row['Song'].replace("'", ""))
    cursor.execute(query)

connection.commit()

# close sql connection
cursor.close()
connection.close()

