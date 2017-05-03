"""
Data accessor utility file

- Gets the data from SQL and forms into a db

- Call 'get_all_data_sets' to get (train_X, train_Y, train_le, test_X, test_Y, test_le) for dataset that has genre

- Call 'get_data_sets_w_lyrics' to get (train_X, train_Y, train_le, test_X, test_Y, test_le) for dataset that has genre AND lyrics

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

from sklearn import preprocessing


#-------------------------
#	Globals
#-------------------------

#-------------------------
#	Functions
#-------------------------

# Get the raw data w/lyrics
def get_all_data_w_lyrics_raw():
	q = """SELECT songs.*, genres.genre
		FROM songs
		LEFT JOIN genres ON genres.songID = songs.track_id
		WHERE genres.genre IS NOT NULL
		AND genres.genre NOT LIKE 'NULL'
		AND songs.has_lyrics = 1
		ORDER BY genres.genre ASC, songs.id ASC;"""

	return mysql_util.execute_dict_query(q)

# Get the raw data
def get_all_data_raw():
	q = """SELECT songs.*, genres.genre
		FROM songs
		LEFT JOIN genres ON genres.songID = songs.track_id
		WHERE genres.genre IS NOT NULL
		AND genres.genre NOT LIKE 'NULL'
		ORDER BY genres.genre ASC, songs.id ASC;"""

	return mysql_util.execute_dict_query(q)

# Remove fields from feature set
def prep_data_set(data):
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
		d.pop('is_train_test', None)
		d.pop('danceability', None)

		results.append(d)

	return results

# Preprocess train and test
def preprocess_data(train_data, test_data):
	train_df = pd.DataFrame(train_data)
	test_df = pd.DataFrame(test_data)

	# Separate target values
	(train_X, train_Y, train_le) = separate_target_values(train_df)
	(test_X, test_Y, test_le) = separate_target_values(test_df)

	# Standardized x values
	column_headers = list(train_X.columns.values)
	scaler = preprocessing.StandardScaler()
	train_X = pd.DataFrame(scaler.fit_transform(train_X))
	test_X = pd.DataFrame(scaler.transform(test_X))
	train_X.columns = column_headers
	test_X.columns = column_headers

	return (train_X, train_Y, train_le, test_X, test_Y, test_le)

# Separate into X & Y values
def separate_target_values(df):
	X = df.drop(['genre'], axis = 1)

	Y = df['genre']

	# Label encodes Y values
	le = preprocessing.LabelEncoder()
	Y = pd.DataFrame(le.fit_transform(Y))
	
	return (X, Y, le)

# 
# Split the data into test and training sets for all data
# 
# +-----------------------+----------+
# | genre                 | count(*) |
# +-----------------------+----------+
# | alternative or folk   |    73483 |
# | classic rock or pop   |    36075 |
# | country               |    18114 |
# | dance and electronica |    81183 |
# | hip-hop/soul/r&b      |    30434 |
# | metal                 |    18939 |
# | pop                   |    49656 |
# +-----------------------+----------+
# 
# TRAIN SIZE : 15K, 25K, 30K
# TEST SIZE : 3K
# 
def split_all_data_sets(data):
	train_data = []
	test_data = []

	alternative = 			[x for x in data if x['genre'] == "alternative or folk"]
	classic_rock_pop = 		[x for x in data if x['genre'] == "classic rock or pop"]
	country = 				[x for x in data if x['genre'] == "country"]
	dance_and_electronica = [x for x in data if x['genre'] == "dance and electronica"]
	hip_hop = 				[x for x in data if x['genre'] == "hip-hop/soul/r&b"]
	metal = 				[x for x in data if x['genre'] == "metal"]
	pop = 					[x for x in data if x['genre'] == "pop"]

	# Train
	train_data.extend(alternative[:30000])
	train_data.extend(classic_rock_pop[:30000])
	train_data.extend(country[:15000])
	train_data.extend(dance_and_electronica[:30000])
	train_data.extend(hip_hop[:25000])
	train_data.extend(metal[:15000])
	train_data.extend(pop[:30000])

	# Test
	test_data.extend(alternative[30000:33000])
	test_data.extend(classic_rock_pop[30000:33000])
	test_data.extend(country[15000:18000])
	test_data.extend(dance_and_electronica[30000:33000])
	test_data.extend(hip_hop[25000:28000])
	test_data.extend(metal[15000:18000])
	test_data.extend(pop[30000:33000])

	print len(train_data)
	print len(test_data)

	# for d in train_data:
	# 	q = "UPDATE songs set is_train_test = 1 where id = %d"%(d['id'])
	# 	mysql_util.execute_query(q)

	# for d in test_data:
	# 	q = "UPDATE songs set is_train_test = 1 where id = %d"%(d['id'])
	# 	mysql_util.execute_query(q)
	
	return preprocess_data(train_data, test_data)

# 
# Split the data into test and training set for data with lyrics
# 
# +-----------------------+----------+
# | genre                 | count(*) |
# +-----------------------+----------+
# | alternative or folk   |    25810 |
# | classic rock / metal  |    20003 |
# | dance and electronica |    21587 |
# | pop                   |    22363 |
# +-----------------------+----------+
# 
# TRAIN SIZE : 17K
# TEST SIZE : 3K
# 
def split_data_sets_w_lyrics(data):
	train_data = []
	test_data = []

	alternative = 			[x for x in data if x['genre'] == "alternative or folk"]
	classic_rock_metal = 	[x for x in data if x['genre'] == "classic rock or pop" or x['genre'] == "metal"]
	dance_and_electronica = [x for x in data if x['genre'] == "dance and electronica"]
	pop = 					[x for x in data if x['genre'] == "pop"]


	# Train
	train_data.extend(alternative[:17000])
	train_data.extend(classic_rock_metal[:17000])
	train_data.extend(dance_and_electronica[:17000])
	train_data.extend(pop[:17000])

	# Test
	test_data.extend(alternative[17000:20000])
	test_data.extend(classic_rock_metal[17000:20000])
	test_data.extend(dance_and_electronica[17000:20000])
	test_data.extend(pop[17000:20000])

	return preprocess_data(train_data, test_data)

# 
# Get all data sets
# 
# returns (train_X, train_Y, train_le, test_X, test_Y, test_le)
# 
def get_all_data_sets():

	# get data
	data = get_all_data_raw()

	# prep data
	results = prep_data_set(data)

	# split into training and test
	return split_all_data_sets(results)

# 
# Get the data w lyrics
# 
# returns (train_X, train_Y, train_le, test_X, test_Y, test_le)
# 
def get_data_sets_w_lyrics():

	# get data
	data = get_all_data_w_lyrics_raw()

	# prep data
	results = prep_data_set(data)

	# split into training and test
	return split_data_sets_w_lyrics(results)

# Get the data and return a dataframee
def get_all_data():

    data = get_all_data_raw()

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
        d.pop('is_train_test', None)
        d.pop('danceability', None)

        results.append(d)

    # print "\n\n"
    df = pd.DataFrame(results)
    # print df

    return df

if __name__=="__main__":
    print get_data_sets_w_lyrics()