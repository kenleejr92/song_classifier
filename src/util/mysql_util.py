"""
MySql utility file


*** NOTE ***

- MUST RUN 'sudo pip install MySQL-python' to install file
- If getting error 'Import Error....Reason: image not found' type 'sudo ln -s /usr/local/mysql/lib/libmysqlclient.18.dylib /usr/local/lib/libmysqlclient.18.dylib'

@author - Tim Mahler
"""

#-------------------------
# Libs
#-------------------------

# External libs
import MySQLdb
import os, sys
import settings


#-------------------------
#	SQL environments
#-------------------------

REMOTE = False	### CHANGE TO SWITCH BETWEEN LOCAL AND REMOTE


HOST_LOCAL = "localhost"
HOST_REMOTE = "54.209.128.159"
HOST = HOST_REMOTE if REMOTE else HOST_LOCAL

PASSWORD_LOCAL = settings.sql_password
PASSWORD = PASSWORD_REMOTE if REMOTE else PASSWORD_LOCAL

DB = "songs"


#-------------------------
#	Functions
#-------------------------
"""
make the connection to MySQL and
execute the query to get mysql data
"""
def execute_query(query):
	connection = MySQLdb.connect(host=HOST, user='root', passwd="root", db=DB, port=3306)
	cursor = connection.cursor()
	cursor.execute(query)
	data = cursor.fetchall()
	cursor.close()
	connection.commit()
	connection.close()
	return data

def execute_dict_query(query):
	connection = MySQLdb.connect(host=HOST, user='root', passwd="root", db=DB, port=3306)
	cursor = connection.cursor(MySQLdb.cursors.DictCursor)
	cursor.execute(query)
	data = cursor.fetchall()
	cursor.close()
	connection.commit()
	connection.close()
	return data