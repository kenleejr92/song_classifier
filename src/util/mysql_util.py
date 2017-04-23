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

HOST = "localhost:"
DB = "db"


#-------------------------
#	Functions
#-------------------------
"""
make the connection to MySQL and
execute the query to get mysql data
"""
def execute_query(query):
	connection = MySQLdb.connect(host=HOST, user='root', passwd="password", db=DB)
	cursor = connection.cursor()
	cursor.execute(query)
	data = cursor.fetchall()
	cursor.close()
	connection.commit()
	connection.close()
	return data

def execute_dict_query(query, db):
	connection = MySQLdb.connect(host=HOST, user='root', passwd="password", db=DB)
	cursor = connection.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute(query)
	data = cursor.fetchall()
	cursor.close()
	connection.commit()
	connection.close()
	return data