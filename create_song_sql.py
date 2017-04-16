"""
Create an SQL table for the Million Song Database
	- converts NFD text to utf8
@author - Alan
"""

import pymysql.cursors
import pandas as pd
import numpy as np
import unicodedata

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
artist VARCHAR(125) DEFAULT NULL,
title VARCHAR(125) DEFAULT NULL,
INDEX artist (artist)
);'''
cursor.execute(sql)
connection.commit()

# populate sql table with data
for index, row in data.iterrows():

	artist_name = "".join(c for c in unicodedata.normalize('NFD', unicode(row['Artist'].decode("utf8"))) if unicodedata.category(c) != "Mn")
	song_name = "".join(c for c in unicodedata.normalize('NFD', unicode(row['Song'].decode("utf8"))) if unicodedata.category(c) != "Mn")

	query = "INSERT INTO songs (songID, artist, title) VALUES ('%s', '%s', '%s')" % (row['ID'], artist_name.replace("'", ""), song_name.replace("'", ""))
	cursor.execute(query)

connection.commit()

# close sql connection
cursor.close()
connection.close()
