import glob
import hdf5_getters
import csv
from tqdm import tqdm
import settings

DEBUG = 1

mylist = dir(hdf5_getters)

print settings.msds_path

#Change this path to wherever you have saved the subset
path_to_data_folder = settings.msds_path

def add_sample_to_table(sample):
	#sample is a list of all the features of a song
	#Your code goes here
	print sample

if DEBUG == 1:
	glob_path = path_to_data_folder + '/data/A/A/A/*'
else:
	glob_path = path_to_data_folder + '/data/*/*/*/*'

filepaths = glob.glob(glob_path)
for filepath in tqdm(filepaths):
	h5 = hdf5_getters.open_h5_file_read(filepath)
	n = hdf5_getters.get_num_songs(h5)
	# print n
	for row in range(n):
		info = []
		for i in range(5,60):
			if i == 35:
				continue
			myfunc = "feature = hdf5_getters." + mylist[i] + "(h5,songidx=row)"
			exec myfunc
			info.append(feature)
		add_sample_to_table(info)

	h5.close()


