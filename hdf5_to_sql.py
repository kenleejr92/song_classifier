'''
create sql dataset from hdf5 files directly
    this is based on Andrew's and Alen's code
@author - Farzan Memarian

'''

import hdf5_getters
from tqdm import tqdm
import pymysql.cursors
import pandas as pd
import numpy as np
import glob
import csv
import pdb

# establish connection to sql server
connection = pymysql.connect(host='localhost',\
   user='root',db='songs')
cursor = connection.cursor()

# create sql table (only need to do this once)
sql = '''CREATE TABLE songs (
songID VARCHAR(50) PRIMARY KEY, 
artist VARCHAR(400) DEFAULT NULL,
title VARCHAR(400) DEFAULT NULL,
INDEX artist (artist)
);'''
cursor.execute(sql)
connection.commit()

glob_path = '/home/frank/1CSEM/1UTCoursesTaken/dataMiningEE380L/termProject/dataset/MillionSongSubset/data/*/*/*/*'
filepaths = glob.glob(glob_path)
for filepath in tqdm(filepaths):
    h5 = hdf5_getters.open_h5_file_read(filepath)
    n = hdf5_getters.get_num_songs(h5)
    # pdb.set_trace()
    # print n
    for row in range(n):
        artist = hdf5_getters.get_artist_name(h5,songidx=row).decode('UTF-8')
        song_id = hdf5_getters.get_song_id(h5,songidx=row).decode('UTF-8')
        title= hdf5_getters.get_title(h5,songidx=row).decode('UTF-8')
        info = [song_id, title, artist]
        # populate sql table with data
        query = "INSERT INTO songs (songID, artist, title) VALUES ('%s', '%s', '%s')"\
                                         % (song_id, artist.replace("'",""), title.replace("'",""))
        cursor.execute(query)
        connection.commit()
        
    h5.close()

# close sql connection
cursor.close()
connection.close()
    
    