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
import re
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
import requests, json
import pandas as pd
from tqdm import tqdm
import csv

#-------------------------
# Globals
#-------------------------
headers = {'Accept-Encoding': 'identity'}

#-------------------------
# Functions
#-------------------------

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

# Main function
def main():
	print "Scraping lyrics"

	myfile = open('lyrics.csv', 'wb')
	wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

	wr.writerow(['id','lyrics'])

	info = pd.read_csv('info.csv', header = 0, names = ['id', 'title','artist'])

	successes = 0

	for index, row in tqdm(info.iterrows()):
		# 1.) Get song lyrics
		genius_song_lyrics = scrape_genius_lyrics(row['artist'], row['title'])
		if genius_song_lyrics != None:
			successes += 1
			wr.writerow([row['id'],genius_song_lyrics.encode('utf-8')])
		else:
			wr.writerow([row['id']])

	print "Finished sraping lyrics. Lyrics found for " + str(float(successes)/float(10000))*100 + " percent of songs."




if __name__=="__main__":
    main()