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
import math

from sklearn import preprocessing
from sklearn.utils import shuffle


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
		d.pop('artist_id', None)
		d.pop('artist_name', None)
		d.pop('song_title', None)
		d.pop('track_id', None)
		d.pop('energy', None)
		d.pop('is_train_test', None)
		d.pop('danceability', None)

		results.append(d)

	return results

# Remove fields from feature set
def prep_lyrics_data_set(data):
	results = []
	for d in data:
		d.pop('id', None)
		d.pop('has_lyrics', None)
		d.pop('deleted', None)
		d.pop('artist_id', None)
		d.pop('artist_name', None)
		d.pop('song_title', None)
		d.pop('track_id', None)
		d.pop('energy', None)
		d.pop('is_train_test', None)
		d.pop('danceability', None)

		results.append(d)

	return results

# Decadize years
def convert_years_to_decade(data):

	for index, row in data.iterrows():
		row['year'] = int(math.floor(int(row['year']) / 10.0)) * 10

	print data

# One hot encode date
def one_hot_encode(data):
	enc = preprocessing.OneHotEncoder()

	vals = enc.fit_transform(data).toarray()

	print vals

	return vals

# Drop column
def drop_column(key, data_set):
	vals = data_set[key]

	data_set = data_set.drop(key, axis=1)

	return (vals, data_set)

# Add column
def add_column(key, vals, data_set):
	data_set.insert(0, key, vals)

	return data_set

# Preprocess train and test
def preprocess_data(train_data, test_data):
	train_df = pd.DataFrame(train_data)
	test_df = pd.DataFrame(test_data)

	# Shuffle
	train_df = shuffle(train_df)
	test_df = shuffle(test_df)

	# Separate target values
	(train_X, train_Y, train_le) = separate_target_values(train_df)
	(test_X, test_Y, test_le) = separate_target_values(test_df)

	# Drop columns
	(train_X_songs_ids, train_X) = drop_column('songID', train_X)
	(test_X_songs_ids, test_X) = drop_column('songID', test_X)
	(train_X_years, train_X) = drop_column('year', train_X)
	(test_X_years, test_X) = drop_column('year', test_X)
	(train_X_musicalKeys, train_X) = drop_column('musicalKey', train_X)
	(test_X_musicalKeys, test_X) = drop_column('musicalKey', test_X)

	# Standardized x values
	column_headers = list(train_X.columns.values)
	scaler = preprocessing.StandardScaler()
	train_X = pd.DataFrame(scaler.fit_transform(train_X))
	test_X = pd.DataFrame(scaler.transform(test_X))
	train_X.columns = column_headers
	test_X.columns = column_headers


	# Re-add columns back
	train_X = add_column("year", train_X_years, train_X)
	test_X = add_column("year", test_X_years, test_X)

	train_X = add_column("musicalKey", train_X_musicalKeys, train_X)
	test_X = add_column("musicalKey", test_X_musicalKeys, test_X)

	train_X = add_column("songID", train_X_songs_ids, train_X)
	test_X = add_column("songID", test_X_songs_ids, test_X)

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

	# print len(train_data)
	# print len(test_data)

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
	results = prep_lyrics_data_set(data)

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

# Convert train and test sets to numpy
def convert_data_sets_to_numpy(train_X, train_Y, test_X, test_Y):

	# drop song id
	train_X = train_X.drop('songID', axis=1)
	test_X = test_X.drop('songID', axis=1)

	# Convert all to matrices
	train_X = train_X.as_matrix()
	train_Y = train_Y.as_matrix()
	test_X = test_X.as_matrix()
	test_Y = test_Y.as_matrix()

	# Reshape
	train_Y = train_Y.reshape((train_Y.shape[0],))
	test_Y = test_Y.reshape((test_Y.shape[0],))

	return (train_X, train_Y, test_X, test_Y)

if __name__=="__main__":
   	print get_data_sets_w_lyrics()
