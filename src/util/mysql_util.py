"""
MySql utility file

@author - Tim Mahler
"""

#-------------------------
# Libs
#-------------------------

# External libs
import MySQLdb
import os, sys


#-------------------------
#	SQL environments
#-------------------------

REMOTE = True	### CHANGE TO SWITCH BETWEEN LOCAL AND REMOTE


HOST_LOCAL = "localhost:"
HOST_REMOTE = "ec2-54-209-131-130.compute-1.amazonaws.com:3306"
HOST = HOST_REMOTE if REMOTE else LOCAL


DB = "songs"


#-------------------------
#	Functions
#-------------------------
"""
make the connection to MySQL and
execute the query to get mysql data
"""
def execute_query(query):
	connection = MySQLdb.connect(host=HOST, user='root', passwd="root", db=DB)
	cursor = connection.cursor()
	cursor.execute(query)
	data = cursor.fetchall()
	cursor.close()
	connection.commit()
	connection.close()
	return data

def execute_dict_query(query, db):
	connection = MySQLdb.connect(host=HOST, user='root', passwd="root", db=DB)
	cursor = connection.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute(query)
	data = cursor.fetchall()
	cursor.close()
	connection.commit()
	connection.close()
	return data