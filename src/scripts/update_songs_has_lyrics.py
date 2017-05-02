'''
updates the lyrics if the file exists
   
@author - Tim Mahler
'''

# in this cell, some of the features of MSD dataset are imported and stored in SQL table. 

#-------------------------
# Libs
#-------------------------

import os, sys, time
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
from util import mysql_util, hdf5_getters, settings, data_accessor_util

#-------------------------
# Globals
#-------------------------

#-------------------------
# Functions
#-------------------------

# Importing the data to sql
def update_songs():
    data = data_accessor_util.get_all_data_raw()

    print "Got data, getting lyric info"

    # For each song, see if lyric file exists
    count = 0
    for d in data:

        file_path = "/home/ubuntu/lyrics/"+d['songID']+".lyrics"

        if os.path.isfile(file_path):
            has_lyrics = 1
        else:
            has_lyrics = 0

        #print "%s has_lyrics = %d"%(file_path, has_lyrics)

        q = """UPDATE songs set has_lyrics = %d WHERE id = %d"""%(has_lyrics, d['id'])
        mysql_util.execute_query(q)

        count = count + 1
        if count % 5000 == 0:
            print "COUNT = %d"%(count)


# Main Script
def main():
    # Update if the songs have lyrics or not
    update_songs()





if __name__=="__main__":
    main()