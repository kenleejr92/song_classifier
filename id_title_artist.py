import glob
import hdf5_getters
import csv
from tqdm import tqdm

myfile = open('info.csv', 'wb')
wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)

glob_path = '/Users/andrew/Documents/datamining/Project/MillionSongSubset/data/*/*/*/*'
filepaths = glob.glob(glob_path)
for filepath in tqdm(filepaths):
	h5 = hdf5_getters.open_h5_file_read(filepath)
	n = hdf5_getters.get_num_songs(h5)
	# print n
	for row in range(n):
		artist = hdf5_getters.get_artist_name(h5,songidx=row)
		song_id = hdf5_getters.get_song_id(h5,songidx=row)
		title= hdf5_getters.get_title(h5,songidx=row)
		info = [song_id, title, artist]
		wr.writerow(info)
	h5.close()



import hdf5_getters
h5 = hdf5_getters.open_h5_file_read(path to some file)
duration = hdf5_getters.get_duration(h5)
h5.close()

