'''
import tags from lastfm dataset
   
@author - Farzan Memarian
 this is based on Andrew's id_title_artist.py and Alen's create_song_sql.py
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

top_tags = { 
0: ['classic rock','classic pop'],
1: ['classical'],
2: ['dance', 'electronica', 'electronic'],
3: ['folk'],
4: ['hip-hop','hiphop','hip-Hop','hip hop'],
5: ['jazz'],
6: ['metal'],
7: ['pop'],
8: ['rock', 'indie'],
9: ['soul', 'reggae'] }

def pick_top_tags(track_tags, top_tags, tags_count):
    match = []
    for key, value in top_tags.items():
        for i,t in enumerate(track_tags):
            temp = re.findall(r"(?=("+'|'.join(value)+r"))",t,flags=re.IGNORECASE)
            if len(temp) == 0:
                continue              
            match.append((key, tags_count[i]))
    sorted(match, key=lambda tup: tup[1])
    return match
           

# for now we are just using the smaller version of the dataset corresponding to 10000 tracks
# for development purposes. And the tags are stored at a pd.DataFrame
glob_path = '/home/frank/1CSEM/1UTCoursesTaken/dataMiningEE380L/termProject/dataset/lastfm_subset/*/*/*/*'
filepaths = glob.glob(glob_path)
data_list = []
for filepath in tqdm(filepaths):
    # print (filepath)
    with open(filepath) as data_file:    
        data = json.load(data_file)
        if len(data.get('tags')) != 0:
            tags = data.get('tags')
            print (tags)
            tags_pure = [i[0] for i in tags]
            tags_count = [i[1] for i in tags]
            match = pick_top_tags(tags_pure,top_tags, tags_count)
            row = [data.get('artist'), data.get('track_id'), data.get('title'), data.get('tags')[0]] 
            data_list.append(row)        
        # pprint(data)
df = pd.DataFrame(data_list, columns = ['artist', 'track_id','title','tags'])
df.loc[df['artist'] == 'Michael Jackson']

