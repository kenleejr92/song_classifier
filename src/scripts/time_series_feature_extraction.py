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

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import FeatureExtractionSettings

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import mysql_util, hdf5_getters, settings

#-------------------------
# Globals
#-------------------------
glob_path = settings.msds_path
filepaths = glob.glob(glob_path)
COLUMN_NAMES = ['pitch0__approximate_entropy__m_2__r_07', 'loud0__approximate_entropy__m_2__r_07', 'timbre0__approximate_entropy__m_2__r_07', 'pitch0__approximate_entropy__m_2__r_05', 'loud0__approximate_entropy__m_2__r_05', 'timbre0__approximate_entropy__m_2__r_05', 'pitch0__approximate_entropy__m_2__r_03', 'loud0__approximate_entropy__m_2__r_03', 'timbre0__approximate_entropy__m_2__r_03', 'pitch0__approximate_entropy__m_2__r_01', 'loud0__approximate_entropy__m_2__r_01', 'timbre0__approximate_entropy__m_2__r_01', 'pitch0__approximate_entropy__m_2__r_09', 'loud0__approximate_entropy__m_2__r_09', 'timbre0__approximate_entropy__m_2__r_09', 'pitch0__autocorrelation__lag_1', 'loud0__autocorrelation__lag_1', 'timbre0__autocorrelation__lag_1', 'pitch0__autocorrelation__lag_3', 'loud0__autocorrelation__lag_3', 'timbre0__autocorrelation__lag_3', 'pitch0__autocorrelation__lag_5', 'loud0__autocorrelation__lag_5', 'timbre0__autocorrelation__lag_5', 'pitch0__autocorrelation__lag_9', 'loud0__autocorrelation__lag_9', 'timbre0__autocorrelation__lag_9', 'pitch0__abs_energy', 'loud0__abs_energy', 'timbre0__abs_energy', 'pitch0__autocorrelation__lag_7', 'loud0__autocorrelation__lag_7', 'timbre0__autocorrelation__lag_7', 'pitch0__kurtosis', 'loud0__kurtosis', 'timbre0__kurtosis', 'pitch0__mean_autocorrelation', 'loud0__mean_autocorrelation', 'timbre0__mean_autocorrelation', 'pitch0__autocorrelation__lag_0', 'loud0__autocorrelation__lag_0', 'timbre0__autocorrelation__lag_0', 'pitch0__autocorrelation__lag_2', 'loud0__autocorrelation__lag_2', 'timbre0__autocorrelation__lag_2', 'pitch0__autocorrelation__lag_4', 'loud0__autocorrelation__lag_4', 'timbre0__autocorrelation__lag_4', 'pitch0__autocorrelation__lag_6', 'loud0__autocorrelation__lag_6', 'timbre0__autocorrelation__lag_6', 'pitch0__autocorrelation__lag_8', 'loud0__autocorrelation__lag_8', 'timbre0__autocorrelation__lag_8', 'pitch0__cwt_coefficients__widths_251020__coeff_0__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_0__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_0__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_0__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_0__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_0__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_0__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_0__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_0__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_0__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_0__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_0__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_1__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_1__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_1__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_1__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_1__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_1__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_1__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_1__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_1__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_1__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_1__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_1__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_2__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_2__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_2__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_2__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_2__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_2__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_2__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_2__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_2__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_2__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_2__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_2__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_3__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_3__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_3__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_3__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_3__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_3__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_3__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_3__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_3__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_3__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_3__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_3__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_4__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_4__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_4__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_4__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_4__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_4__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_4__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_4__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_4__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_4__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_4__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_4__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_5__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_5__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_5__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_5__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_5__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_5__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_5__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_5__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_5__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_5__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_5__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_5__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_6__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_6__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_6__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_6__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_6__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_6__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_6__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_6__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_6__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_6__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_6__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_6__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_7__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_7__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_7__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_7__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_7__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_7__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_7__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_7__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_7__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_7__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_7__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_7__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_8__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_8__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_8__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_8__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_8__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_8__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_8__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_8__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_8__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_8__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_8__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_8__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_9__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_9__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_9__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_9__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_9__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_9__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_9__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_9__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_9__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_9__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_9__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_9__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_10__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_10__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_10__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_10__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_10__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_10__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_10__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_10__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_10__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_10__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_10__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_10__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_11__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_11__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_11__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_11__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_11__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_11__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_11__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_11__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_11__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_11__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_11__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_11__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_12__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_12__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_12__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_12__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_12__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_12__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_12__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_12__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_12__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_12__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_12__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_12__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_13__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_13__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_13__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_13__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_13__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_13__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_13__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_13__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_13__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_13__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_13__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_13__w_20', 'pitch0__cwt_coefficients__widths_251020__coeff_14__w_2', 'loud0__cwt_coefficients__widths_251020__coeff_14__w_2', 'timbre0__cwt_coefficients__widths_251020__coeff_14__w_2', 'pitch0__cwt_coefficients__widths_251020__coeff_14__w_5', 'loud0__cwt_coefficients__widths_251020__coeff_14__w_5', 'timbre0__cwt_coefficients__widths_251020__coeff_14__w_5', 'pitch0__cwt_coefficients__widths_251020__coeff_14__w_10', 'loud0__cwt_coefficients__widths_251020__coeff_14__w_10', 'timbre0__cwt_coefficients__widths_251020__coeff_14__w_10', 'pitch0__cwt_coefficients__widths_251020__coeff_14__w_20', 'loud0__cwt_coefficients__widths_251020__coeff_14__w_20', 'timbre0__cwt_coefficients__widths_251020__coeff_14__w_20']
#-------------------------
# Functions
#-------------------------
def init_tables():
    print "Initializing tables"

    # Drop database
    q = """DROP TABLE IF EXISTS time_series_features;"""
    mysql_util.execute_query(q)

    # Create table schema
    sql = """CREATE TABLE IF NOT EXISTS time_series_features (

        track_id VARCHAR(64),
        segments_length INT,
        deleted tinyint(1) DEFAULT 0,
        has_lyrics tinyint(1) DEFAULT 0,
        PRIMARY KEY (track_id)
    );"""
    print sql
    mysql_util.execute_query(sql)
    for c in COLUMN_NAMES:
        sql = """ALTER TABLE time_series_features ADD COLUMN %s REAL;""" % c
        mysql_util.execute_query(sql)

# Importing the data to sql
def import_data_to_sql():
    # Check to make sure valid inputs
    feature_key_words = ['cwt_coefficients','percentage_of_reoccuring_values_to_all_values',\
    'autocorrelation','kurtosis','abs_energy','approximate_entropy']
    extraction_settings = FeatureExtractionSettings()
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
                        feature_vector = []
                        song_id = hdf5_getters.get_song_id(h5,songidx=row).decode('UTF-8')
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

                        #time series features
                        #perfom feature extraction
                        max_loudness_time = hdf5_getters.get_segments_loudness_max_time(h5,songidx=row)
                        segments_pitches = hdf5_getters.get_segments_pitches(h5,songidx=row)
                        segments_timbre = hdf5_getters.get_segments_timbre(h5,songidx=row)
                        max_loudness = hdf5_getters.get_segments_loudness_max(h5,songidx=row)
                        segments_length = segments_pitches.shape[0]
                        segments_pitches = np.mean(segments_pitches,axis=1)
                        segments_timbre = np.mean(segments_timbre,axis=1)
                        max_loudness = pd.DataFrame(max_loudness)
                        max_loudness['id'] = 0
                        segments_pitches = pd.DataFrame(segments_pitches)
                        segments_pitches['id']=0
                        segments_timbre = pd.DataFrame(segments_timbre)
                        segments_timbre['id']=0
                        
                        ml_features = extract_features(max_loudness,column_id='id',feature_extraction_settings=extraction_settings)
                        sp_features = extract_features(segments_pitches,column_id='id',feature_extraction_settings=extraction_settings)
                        st_features = extract_features(segments_timbre,column_id='id',feature_extraction_settings=extraction_settings)

                        for feature in sp_features:
                            for key_word in feature_key_words:
                                if key_word in feature:
                                    feature_vector.append(sp_features[feature][0])
                                    feature_vector.append(st_features[feature][0])
                                    feature_vector.append(ml_features[feature][0])
                                    # feature = feature.replace(".","")
                                    # feature = feature.replace("(","")
                                    # feature = feature.replace(")","")
                                    # feature = feature.replace(" ","")
                                    # feature = feature.replace(",","")
                                    # COLUMN_NAMES.append('pitch'+feature)
                                    # COLUMN_NAMES.append('loud'+feature)
                                    # COLUMN_NAMES.append('timbre' + feature) 
                        #for idx,val in enumerate(l): 
                         #   if type(val)==np.float64 and np.isnan(val):
                          #      l[idx]=0. 
                        meta_features = [str(track_id), str(segments_length)]
                        meta_names = ['track_id', 'segments_length']
                        insert_query = 'INSERT INTO time_series_features ('
                        for m in meta_names:
                            insert_query = insert_query + m + ','
                        for c in COLUMN_NAMES[:-1]:
                            insert_query = insert_query + c + ','
                        insert_query = insert_query + COLUMN_NAMES[-1] + ') VALUES ('
                        for l in meta_features[:-1]:
                            insert_query = insert_query + "'%s'," % l
                        insert_query = insert_query + meta_features[-1] + ','
                        for v in feature_vector[:-1]:
                            insert_query = insert_query + str(v) + ','
                        insert_query = insert_query + str(feature_vector[-1]) + ');'
                        insert_query = """%s""" % insert_query
                        mysql_util.execute_query(insert_query)

                    h5.close()

# Main Script
def main():
    # 1.) Init tables
    init_tables()

    # 2.) Import data from songs hdf5 files to sql
    import_data_to_sql()





if __name__=="__main__":
    main()
