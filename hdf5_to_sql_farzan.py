
# coding: utf-8

# In[ ]:

'''
create sql dataset from hdf5 files directly
   
@author - Farzan Memarian
 this is based on Andrew's id_title_artist.py and Alen's create_song_sql.py
'''

# in this cell, some of the features of MSD dataset are imported and stored in SQL table. 


import hdf5_getters
from tqdm import tqdm
import pymysql.cursors
import pandas as pd
import numpy as np
import glob
import csv
import pdb
import json
from pprint import pprint
import unicodedata

glob_path = '/home/frank/1CSEM/1UTCoursesTaken/dataMiningEE380L/termProject/teamproject/dataset/MillionSongSubset/data/*/*/*/*'
filepaths = glob.glob(glob_path)

# establish connection to sql server
connection = pymysql.connect(host='localhost',   user='root',db='songs')
cursor = connection.cursor()

# create sql table (only need to do this once)
sql = '''CREATE TABLE songs (

songID VARCHAR(50) PRIMARY KEY, 
artist VARCHAR(400) DEFAULT NULL,
title VARCHAR(400) DEFAULT NULL,
danceability REAL,
duration REAL,
energy REAL,
loudness REAL,
musicalKey INT,
mode INT,
tempo REAL,

INDEX artist (artist)
);'''
cursor.execute(sql)
connection.commit()


glob_path = '/home/frank/1CSEM/1UTCoursesTaken/dataMiningEE380L/termProject/dataset/MillionSongSubset/data/*/*/*/*'
filepaths = glob.glob(glob_path)
for filepath in tqdm(filepaths):
    h5 = hdf5_getters.open_h5_file_read(filepath)
    n = hdf5_getters.get_num_songs(h5)
    # print n
    for row in range(n):
        artist = hdf5_getters.get_artist_name(h5,songidx=row)#.decode('UTF-8')
        song_id = hdf5_getters.get_song_id(h5,songidx=row).decode('UTF-8')
        title= hdf5_getters.get_title(h5,songidx=row)#.decode('UTF-8')
        artist = "".join(c for c in unicodedata.normalize('NFD', str(artist.decode("utf8"))) if unicodedata.category(c) != "Mn")
        title = "".join(c for c in unicodedata.normalize('NFD', str(title.decode("utf8"))) if unicodedata.category(c) != "Mn")
        danceability = hdf5_getters.get_danceability(h5,songidx=row)
        duration = hdf5_getters.get_duration(h5,songidx=row)
        energy = hdf5_getters.get_energy(h5,songidx=row)
        loudness = hdf5_getters.get_loudness(h5,songidx=row)
        musicalKey = hdf5_getters.get_key(h5,songidx=row)
        mode = hdf5_getters.get_mode(h5,songidx=row)
        tempo = hdf5_getters.get_tempo(h5,songidx=row)
        # artist_mbtags = hdf5_getters.get_artist_mbtags(h5,songidx=row).astype('U13')
        # artist_mbtags_count = hdf5_getters.get_artist_mbtags_count(h5,songidx=row)
        # artist_terms = hdf5_getters.get_artist_terms(h5, songidx=row).astype('U13')
#         beats_start_temp = hdf5_getters.get_beats_start(h5,songidx=row)
#         beats_start = beats_start_temp[-1] / len(beats_start_temp)
#         segments_loudness_max_temp = hdf5_getters.get_segments_loudness_max(h5,songidx=row)
#         segments_loudness_max = np.mean(segments_loudness_max_temp)
#         bars_start_temp = hdf5_getters.get_bars_start(h5,songidx=row)
#         bars_start = bars_start_temp[-1] / len(bars_start_temp)
#         end_of_fade_in = hdf5_getters.get_end_of_fade_in(h5,songidx=row)
#         segments_loudness_max_time_temp = hdf5_getters.get_segments_loudness_max_time(h5,songidx=row)
#         segments_loudness_max_time = np.mean(segments_loudness_max_time_temp)
#         sections_start_temp = hdf5_getters.get_sections_start(h5,songidx=row)
#         sections_start = sections_start_temp[-1] / len(sections_start_temp) 
#         segments_pitches_temp = hdf5_getters.get_segments_pitches_temp(h5,songidx=row)
#         segments_pitches = np.mean(segments_pitches_temp)
#         segments_timbre_temp = hdf5_getters.get_segments_timbre_temp(h5,songidx=row)
#         segments_timbre = np.mean(segments_timbre_temp)        
#         tatums_start_temp = hdf5_getters.get_tatums_start(h5,songidx=row)
#         tatums_start = tatums_start_temp[-1] / len(tatums_start_temp)
#         start_of_fade_out = hdf5_getters.get_start_of_fade_out(h5,songidx=row)
        
        
        # print (artist)
        # print (title)
        # print (song_id)
        # print (artist_terms)
        # print(tempo)
        # print(artist_mbtags)
        # print(artist_mbtags_count)
        # pdb.set_trace()
        # populate sql table with data
        
        query = "INSERT INTO songs (songID, artist, title, danceability, duration, energy, loudness, musicalKey,            mode, tempo) VALUES ('%s', '%s','%s','%s','%s','%s','%s','%s','%s','%s')"            % (song_id, 
               artist.replace("'",""), 
               title.replace("'",""), 
               danceability, 
               duration, 
               energy,
               loudness, 
               musicalKey, 
               mode, 
               tempo)
        cursor.execute(query)
        connection.commit()
    h5.close()

# close sql connection
cursor.close()
connection.close()


