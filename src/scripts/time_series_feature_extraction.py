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

#-------------------------
# Functions
#-------------------------
 
# Importing the data to sql
def import_data_to_sql():
    # Check to make sure valid inputs
    feature_key_words = ['cwt_coefficients','percentage_of_reoccuring_values_to_all_values',\
    'autocorrelation','kurtosis','fft_coefficient','abs_energy','number_peaks',approximate_entropy']
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
                        max_loudness = pd.DataFrame(max_loudness)
                        max_loudness['id'] = 0
                        segments_pitches = pd.DataFrame(segments_pitches)
                        segments_pitches['id']=0
                        segments_timbre = pd.DataFrame(segments_timbre)
                        segments_timbre['id']=0
                        
                        l = [max_loudness,segments_pitches,segments_timbre]
                        ml_features = extract_features(max_loudness,column_id='id',feature_extraction_settings=extraction_settings)
                        sp_features = extract_features(segments_pitches,column_id='id',feature_extraction_settings=extraction_settings)
                        st_feature = extract_features(segments_timbre,column_id='id',feature_extraction_settings=extraction_settings)
                        print sp_features
                        print sp_features.columns
                        for s in sp_features: print(s)
                        for fkw in feature_key_words:
                            pass
                        #for idx,val in enumerate(l): 
                         #   if type(val)==np.float64 and np.isnan(val):
                          #      l[idx]=0. 

                    h5.close()

# Main Script
def main():
    # 1.) Init tables
    #init_tables()

    # 2.) Import data from songs hdf5 files to sql
    import_data_to_sql()





if __name__=="__main__":
    main()
