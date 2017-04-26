"""
Create an SQL table for the Million Song Database: author and title only
    - converts NFD text to utf8
@author - Alan
"""
import hdf5_getters
import glob
import pymysql.cursors
import pandas as pd
import numpy as np
import unicodedata
from tqdm import tqdm

# read csv file containing id, song, artist
# data = pd.read_csv('./info.csv', delimiter = ",", names=['ID','Song','Artist']);
# data = data.replace(np.nan, '', regex=True)

# read from hdf5 files
glob_path = '/home/ubuntu/song_data/data/*/*/*/*'
filepaths = glob.glob(glob_path)

# establish connection to sql server
connection = pymysql.connect(host='localhost',\
   user='root',password='root',db='songs')

cursor = connection.cursor()

cursor.execute('SET NAMES utf8;')
cursor.execute('SET CHARACTER SET utf8;')
cursor.execute('SET character_set_connection=utf8;')

# create sql table (only need to do this once)
sql = '''CREATE TABLE song_titles (
pkID INT PRIMARY KEY AUTO_INCREMENT,
songID VARCHAR(50) DEFAULT NULL, 
artist VARCHAR(200) DEFAULT NULL,
title VARCHAR(200) DEFAULT NULL,
INDEX songID (songID)
);'''
cursor.execute(sql)
connection.commit()

for filepath in filepaths:
    h5 = hdf5_getters.open_h5_file_read(filepath)
    n = hdf5_getters.get_num_songs(h5)

    for row in range(n):
        artist = hdf5_getters.get_artist_name(h5,songidx=row)
        song_id = hdf5_getters.get_song_id(h5,songidx=row)
        title= hdf5_getters.get_title(h5,songidx=row)

        try:
            song_id = song_id.decode('UTF-8')

            artist = "".join(c for c in unicodedata.normalize('NFD', unicode(artist.decode("utf8"))) if unicodedata.category(c) != "Mn")
            title = "".join(c for c in unicodedata.normalize('NFD', unicode(title.decode("utf8"))) if unicodedata.category(c) != "Mn")
            artist = artist.replace("'", "")
            title = title.replace("'", "")

        except:
            continue



        query = "INSERT INTO song_titles (songID, artist, title) VALUES ('%s', '%s', '%s')" % (song_id, artist, title)
        cursor.execute(query)

    h5.close()


"""
# populate sql table with data, csv version
for index, row in data.iterrows():

    artist_name = "".join(c for c in unicodedata.normalize('NFD', unicode(row['Artist'].decode("utf8"))) if unicodedata.category(c) != "Mn")
    song_name = "".join(c for c in unicodedata.normalize('NFD', unicode(row['Song'].decode("utf8"))) if unicodedata.category(c) != "Mn")

    query = "INSERT INTO songs (songID, artist, title) VALUES ('%s', '%s', '%s')" % (row['ID'], artist_name.replace("'", ""), song_name.replace("'", ""))
    cursor.execute(query)
"""

connection.commit()
# close sql connection
cursor.close()
connection.close()
