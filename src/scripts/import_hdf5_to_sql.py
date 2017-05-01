'''
create sql dataset from hdf5 files directly
   
@author - Ken Lee
 this is based on Andrew's id_title_artist.py and Alen's create_song_sql.py
'''

# in this cell, some of the features of MSD dataset are imported and stored in SQL table. 

#-------------------------
# Libs
#-------------------------

import os, sys
from tqdm import tqdm
import pymysql.cursors
import pandas as pd
import numpy as np
import glob
import csv
import json
from pprint import pprint
import unicodedata
import re
import string

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import mysql_util, hdf5_getters, settings

#-------------------------
# Globals
#-------------------------
glob_path = settings.msds_path
filepaths = glob.glob(glob_path)

#-------------------------
# Functions
#-------------------------

# Init tables
def init_tables():
    print "Initializing tables"

    # Drop database
    q = """DROP TABLE IF EXISTS songs;"""
    mysql_util.execute_query(q)

    # Create table schema
    sql = """CREATE TABLE IF NOT EXISTS songs (

        id INT unsigned NOT NULL AUTO_INCREMENT,
        songID VARCHAR(64), 
        danceability REAL DEFAULT NULL,
        duration REAL DEFAULT NULL,
        energy REAL DEFAULT NULL,
        loudness REAL DEFAULT NULL,
        musicalKey INT DEFAULT NULL,
        mode INT DEFAULT NULL,
        tempo REAL DEFAULT NULL,
        time_signature INT DEFAULT NULL,
        year INT DEFAULT NULL,
        song_hottness REAL DEFAULT NULL,
        max_loudness REAL DEFAULT NULL,
        end_of_fade_in REAL DEFAULT NULL,
        start_of_fade_out REAL DEFAULT NULL,
        bars_start REAL DEFAULT NULL,
        beats_start REAL DEFAULT NULL,
        sections_start REAL DEFAULT NULL,
        tatums_start REAL DEFAULT NULL,
        segments_start REAL DEFAULT NULL,
        max_loudness_time REAL DEFAULT NULL,
        segments_loudness_start REAL DEFAULT NULL,
        segments_pitches REAL DEFAULT NULL,
        segments_timbre REAL DEFAULT NULL,

        artist_id VARCHAR(128),
        artist_name VARCHAR(64),
        song_title VARCHAR(64),
        track_id VARCHAR(64),

        deleted tinyint(1) DEFAULT 0,
        has_lyrics tinyint(1) DEFAULT 0,

        PRIMARY KEY (id)
    );"""
    print sql
    mysql_util.execute_query(sql)
    
# Importing the data to sql
def import_data_to_sql():
    # Check to make sure valid inputs
    if len(sys.argv) < 2:
        print "Must array of alphabaet characters"
        return

    input_letters = sys.argv[1]

    alphas = list(string.ascii_uppercase)

    print alphas

    print input_letters

    # Loop only though alphabet
    for letter in alphas:
        # If letter in input, then loop through
        if letter in input_letters:
                path = settings.base_msds_path.replace("$VAL", letter)
                print path
                filepaths = glob.glob(path)

                # Filepath for only letter named songs
                for filepath in tqdm(filepaths):
                    h5 = hdf5_getters.open_h5_file_read(filepath)
                    n = hdf5_getters.get_num_songs(h5)
                    for row in range(n):
                        song_id = hdf5_getters.get_song_id(h5,songidx=row).decode('UTF-8')
                        print "Converting song %s"%(song_id)
                #         artist = hdf5_getters.get_artist_name(h5,songidx=row).decode('UTF-8')
                #         title= hdf5_getters.get_title(h5,songidx=row)#.decode('UTF-8')
                #         artist = "".join(c for c in unicodedata.normalize('NFD', str(artist.decode("utf8"))) if unicodedata.category(c) != "Mn")
                #         title = "".join(c for c in unicodedata.normalize('NFD', str(title.decode("utf8"))) if unicodedata.category(c) != "Mn")
                        #single number features
                        danceability = hdf5_getters.get_danceability(h5,songidx=row)
                        duration = hdf5_getters.get_duration(h5,songidx=row)
                        energy = hdf5_getters.get_energy(h5,songidx=row)
                        loudness = hdf5_getters.get_loudness(h5,songidx=row)
                        musicalKey = hdf5_getters.get_key(h5,songidx=row)
                        mode = hdf5_getters.get_mode(h5,songidx=row)
                        tempo = hdf5_getters.get_tempo(h5,songidx=row)
                        time_signature = hdf5_getters.get_time_signature(h5,songidx=row)
                        year = hdf5_getters.get_year(h5,songidx=row)
                        song_hottness = hdf5_getters.get_song_hotttnesss(h5,songidx=row)
                        end_of_fade_in = hdf5_getters.get_end_of_fade_in(h5,songidx=row)
                        start_of_fade_out = hdf5_getters.get_start_of_fade_out(h5,songidx=row)

                        artist_id = hdf5_getters.get_artist_id(h5, songidx=row)
                        artist_name = hdf5_getters.get_artist_name(h5, songidx=row)
                        song_title = hdf5_getters.get_title(h5, songidx=row)
                        track_id = hdf5_getters.get_track_id(h5, songidx=row)


                        # Parse song name and title
                        artist_name = unicode(artist_name, "utf-8")
                        song_title = unicode(song_title, "utf-8")
                        artist_name = "".join(c for c in unicodedata.normalize('NFD', artist_name) if unicodedata.category(c) != "Mn")
                        song_title = "".join(c for c in unicodedata.normalize('NFD', song_title) if unicodedata.category(c) != "Mn")
                        artist_name = re.sub('[^0-9a-zA-Z\w\s]+', '', artist_name)
                        song_title = re.sub('[^0-9a-zA-Z\w\s]+', '', song_title)

                        
                        #timestamp features
                        #take last element and divide by length to get beats/unit time, segments/unit_time
                        bars_start = hdf5_getters.get_bars_start(h5,songidx=row)
                        beats_start = hdf5_getters.get_beats_start(h5,songidx=row)
                        sections_start = hdf5_getters.get_sections_start(h5,songidx=row)
                        tatums_start = hdf5_getters.get_tatums_start(h5,songidx=row)
                        segments_start = hdf5_getters.get_segments_start(h5,songidx=row)
                        if len(bars_start)==0: bars_start = 0.
                        else: bars_start = bars_start[-1]/len(bars_start)
                        if len(beats_start)==0: beats_start = 0.
                        else: beats_start = beats_start[-1]/len(beats_start)
                        if len(sections_start)==0: sections_start = 0.
                        else: sections_start = sections_start[-1]/len(sections_start)
                        if len(tatums_start)==0: tatums_start = 0.
                        else: tatums_start = tatums_start[-1]/len(tatums_start)
                        if len(segments_start)==0: segments_start = 0.
                        else: segments_start = segments_start[-1]/len(segments_start)
                        
                        #time series features
                        #take mean
                        max_loudness_time = hdf5_getters.get_segments_loudness_max_time(h5,songidx=row)
                        segments_loudness_start = hdf5_getters.get_segments_loudness_start(h5,songidx=row)
                        segments_pitches = hdf5_getters.get_segments_pitches(h5,songidx=row)
                        segments_timbre = hdf5_getters.get_segments_timbre(h5,songidx=row)
                        max_loudness = hdf5_getters.get_segments_loudness_max(h5,songidx=row)
                        segments_pitches = np.mean(segments_pitches)
                        segments_timbre = np.mean(segments_timbre)
                        max_loudness = np.mean(max_loudness)
                        max_loudness_time = np.mean(max_loudness_time)
                        segments_loudness_start = np.mean(segments_loudness_start)
                        
                        l = [song_id, danceability, duration, energy, loudness, musicalKey, mode, 
                             tempo, time_signature, year, song_hottness, max_loudness, end_of_fade_in,
                             start_of_fade_out, bars_start, beats_start, sections_start, tatums_start,
                             segments_start, max_loudness_time, segments_loudness_start, segments_pitches,
                             segments_timbre, artist_id, artist_name, song_title, track_id]
                        
                        for idx,val in enumerate(l): 
                            if type(val)==np.float64 and np.isnan(val):
                                l[idx]=0.
                                
                        insert_query = """INSERT INTO songs (songID, danceability, duration, energy, loudness, musicalKey,
                            mode, tempo, time_signature, year, song_hottness, max_loudness, end_of_fade_in, start_of_fade_out,
                            bars_start, beats_start, sections_start, tatums_start, segments_start, max_loudness_time,
                            segments_loudness_start, segments_pitches, segments_timbre, artist_id, artist_name, song_title, track_id) 
                            VALUES ('%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s','%s',
                            '%s','%s','%s','%s','%s','%s','%s','%s','%s','%s', '%s', '%s', '%s', '%s')"""% tuple(l)
                        mysql_util.execute_query(insert_query)

                    h5.close()


# Main Script
def main():
    # 1.) Init tables
    #init_tables()

    # 2.) Import data from songs hdf5 files to sql
    import_data_to_sql()





if __name__=="__main__":
    main()