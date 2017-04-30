'''
import tags from lastfm dataset
   
@author - Farzan Memarian 
 Read data from lastfm dataset, they come in train and test sets.
 we will read the paths separately and merge the paths. 
'''


from tqdm import tqdm
import pymysql.cursors
import pandas as pd
import numpy as np
import glob
import csv
import pdb
import json
from pprint import pprint
import pdb
import re
import unicodedata
import sys, os

#sys.path.append( '/Users/andrew/Documents/datamining/Project/song_classifier/src/util')
sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
import hdf5_getters, settings

LOCAL = True


top_tags = { 
'classic rock or pop': ['classic rock','classic pop'],
'classical': ['classical'],
'dance and electronica': ['dance', 'electronica', 'electronic'],
'folk': ['folk'],
'hip-hop': ['hip-hop','hiphop','hip-Hop','hip hop'],
'jazz': ['jazz'],
'metal': ['metal'],
'pop': ['pop'],
'rock and indie': ['rock', 'indie'],
'soul and reggae': ['soul', 'reggae'] }

def pick_top_tags(track_tags, top_tags, tags_count):
    match = []
    for key, value in top_tags.items():
        for i,t in enumerate(track_tags):
            temp = re.findall(r"(?=("+'|'.join(value)+r"))",t,flags=re.IGNORECASE)
            if len(temp) == 0:
                continue              
            match.append((key, tags_count[i]))
    sorted(match, key=lambda tup: tup[1])
    if len(match) == 0: return 'NULL'
    return match[0][0]
           
if LOCAL:
    glob_path = settings.lastfm_path
    filepaths = glob.glob(glob_path)
else:    
    glob_path = '/home/ubuntu/lastfm_dataset/lastfm_train/*/*/*/*'
    filepaths_train = glob.glob(glob_path)
    glob_path = '/home/ubuntu/lastfm_dataset/lastfm_test/*/*/*/*'
    filepaths_test = glob.glob(glob_path)
    filepaths = filepaths_train
    filepaths.extend(filepaths_test)
# establish connection to sql server
connection = pymysql.connect(host='localhost',\
   user='root',password=settings.sql_password,db='songs')
cursor = connection.cursor()

# create the sql table if it didn't exits
sql = '''DROP TABLE IF EXISTS genres;'''
cursor.execute(sql)
connection.commit()
sql = '''CREATE TABLE IF NOT EXISTS genres (
songID VARCHAR(50) PRIMARY KEY, 
genre VARCHAR(50) DEFAULT NULL,
INDEX songID (songID)
);'''
cursor.execute(sql)
connection.commit()

for filepath in tqdm(filepaths):
    # print (filepath)
    with open(filepath) as data_file:    
        data = json.load(data_file)
        if len(data.get('tags')) != 0:
            tags = data.get('tags')
            song_id = data.get('track_id')
            tags_pure = [i[0] for i in tags]
            tags_count = [i[1] for i in tags]
            match = pick_top_tags(tags_pure,top_tags, tags_count)
            query = "INSERT INTO genres (songID, genre) VALUES ('%s','%s')" % (song_id, match)
            cursor.execute(query)
            connection.commit()
cursor.close()
connection.close()
