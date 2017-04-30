'''
import tags from lastfm dataset
   
@author - Ken Lee
 Change lastfm location and mysql password
'''

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
import pdb
import re
import unicodedata

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
           

# for now we are just using the smaller version of the dataset corresponding to 10000 tracks
# for development purposes. And the tags are stored at a pd.DataFrame
glob_path = '~/lastfm_dataset/lastfm_train/*/*/*/*'
filepaths_train = glob.glob(glob_path)
import pdb; pdb.set_trace()
glob_path = '~/lastfm_dataset/lastfm_train/*/*/*/*'
filepaths_test = glob.glob(glob_path)
# establish connection to sql server
connection = pymysql.connect(host='localhost',\
   user='root',password='password',db='songs')
cursor = connection.cursor()

# create sql table (only need to do this once)
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