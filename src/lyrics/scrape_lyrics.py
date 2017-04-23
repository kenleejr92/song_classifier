#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
Scrape the lyrics for the datamining project
	- Given an artist name and song name, scrape the lyrics

@author - Tim

"""

#-------------------------
# Libs
#-------------------------

# External libs
import os, sys, time
import unicodedata
import re
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
import requests, json
import pandas as pd
#from tqdm import tqdm
import csv

sys.path.append( os.path.realpath("%s/.."%os.path.dirname(__file__)) )
from util import mysql_util

#-------------------------
# Globals
#-------------------------
headers = {'Accept-Encoding': 'identity'}
azheaders = { 'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0' }


#-------------------------
# Functions
#-------------------------

# Trim artist and song
def trim_song_info(artist_name, song_name):
	artist_name = "".join(c for c in unicodedata.normalize('NFD', artist_name) if unicodedata.category(c) != "Mn")
	song_name = "".join(c for c in unicodedata.normalize('NFD', song_name) if unicodedata.category(c) != "Mn")

	artist_name = re.sub('[^0-9a-zA-Z\w\s]+', '', artist_name)
	song_name = re.sub('[^0-9a-zA-Z\w\s]+', '', song_name)

	return (artist_name, song_name)


# Build genius url
def build_genius_url(artist_name, song_name):
	artist_name = re.sub('[^0-9a-zA-Z\w\s]+', '', artist_name)
	song_name = re.sub('[^0-9a-zA-Z\w\s]+', '', song_name)
	return "https://genius.com/"+artist_name.title().replace(" ", "-")+"-"+song_name.replace(" ", "-")+"-lyrics"

# Scrape genius for lyrics
def scrape_genius_lyrics(artist_name, song_name):
	song_url = build_genius_url(artist_name, song_name)
	print "Song URL = %s\n"%(song_url)
	html = requests.get(song_url, headers=headers)

	# most_recent_list
	doc = BeautifulSoup(html.text, 'html.parser')
	lyrics_div = doc.find_all('lyrics', {"class" : "lyrics"})
	if lyrics_div and len(lyrics_div) > 0:
		lyrics = lyrics_div[0].text
		lyrics = re.sub(r"\[.*\]", "", lyrics).lstrip()	# Remove the brackets from the lyrics and preliminary spaces
		print lyrics
		return lyrics
	else:
		print "ERROR: Lyrics on genius not found!"
		
		return None

def build_az_lryics(artist_name, song_name):
	artist_name = re.sub('[^0-9a-zA-Z\w\s]+', '', artist_name)
	song_name = re.sub('[^0-9a-zA-Z\w\s]+', '', song_name)

	return "http://www.azlyrics.com/lyrics/"+artist_name.replace(" ", "")+"/"+song_name.replace(" ", "")+".html"

# Scrape genius for lyrics
def scrape_az_lyrics(artist_name, song_name):
	song_url = build_az_lryics(artist_name, song_name)
	print "Song URL = %s\n"%(song_url)

	html = requests.get(song_url, headers=azheaders)

	# most_recent_list
	doc = BeautifulSoup(html.text, 'html.parser')
	lyrics_divs = doc.find_all('div')

	idx = 0
	for i, d in enumerate(lyrics_divs):
		if d.get("class") and len(d.get("class")) > 0 and "ringtone" in d.get("class"):
			idx = i
			break
	lyrics_div = lyrics_divs[idx+1]

	print lyrics_div.text
	# if lyrics_div and len(lyrics_div) > 0:
	# 	lyrics = lyrics_div[0].text
	# 	lyrics = re.sub(r"\[.*\]", "", lyrics).lstrip()	# Remove the brackets from the lyrics and preliminary spaces
	# 	print lyrics
	# 	return lyrics
	# else:
	# 	print "ERROR: Lyrics on genius not found!"
		
	# 	return None


# Main function
def get_from_csv():
	print "Scraping lyrics"

	myfile = open('lyrics.csv', 'wb')
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

	wr.writerow(['id','lyrics'])

	info = pd.read_csv('info.csv', header = 0, names = ['id', 'title','artist'])

	successes = 0

	for index, row in tqdm(info.iterrows()):
		# 1.) Get song lyrics
		genius_song_lyrics = scrape_genius_lyrics(row['artist'], row['title'])

		az_song_lyrics = scrape_az_lyrics(row['artist'], row['title'])


		if genius_song_lyrics != None:
			successes += 1
			wr.writerow([row['id'],genius_song_lyrics.encode('utf-8')])
		else:
			wr.writerow([row['id']])

	print "Finished sraping lyrics. Lyrics found for " + str(float(successes)/float(10000))*100 + " percent of songs."

# Main function
def main():
	print "Scraping lyrics"

	artist = u"Flávio José"
	song = u"É Sempre Assim"

	(artist, song) = trim_song_info(artist, song)

	print "Artist = %s"%(artist)
	print "Song = %s"%(song)

	# 1.) Get song lyrics
	print "********************"
	print "Genius\n"
	print "********************"
	genius_song_lyrics = scrape_genius_lyrics(artist, song)

	print "********************"
	print "AZ\n"
	print "********************"
	az_song_lyrics = scrape_az_lyrics(artist, song)


if __name__=="__main__":
    main()