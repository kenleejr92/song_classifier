"""
Data accessor utility file

- Gets the data from SQL and forms into a db

@author - Tim Mahler
"""

#-------------------------
# Libs
#-------------------------

# External libs
import os, sys
import settings
import mysql_util
import pandas as pd
import itertools


#-------------------------
#	Globals
#-------------------------

#-------------------------
#	Functions
#-------------------------

# Get the data and return a dataframee
def get_all_data():
	q = """SELECT songs.*, genres.genre
		FROM songs
		LEFT JOIN genres ON genres.songID = songs.track_id
		WHERE genres.genre IS NOT NULL
		AND genres.genre NOT LIKE 'NULL';"""

	data = mysql_util.execute_dict_query(q)

	# Delete fields we dont want
	results = []
	for d in data:
		d.pop('id', None)
		d.pop('has_lyrics', None)
		d.pop('deleted', None)
		d.pop('songID', None)
		d.pop('artist_id', None)
		d.pop('artist_name', None)
		d.pop('song_title', None)
		d.pop('track_id', None)
		d.pop('energy', None)

		results.append(d)

	# print "\n\n"
	df = pd.DataFrame(results)
	# print df

	return df

if __name__=="__main__":
    get_all_data()