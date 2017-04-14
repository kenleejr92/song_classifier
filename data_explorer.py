import glob
import hdf5_getters
import csv
from tqdm import tqdm


mylist = dir(hdf5_getters)




glob_path = '/Users/andrew/Documents/datamining/Project/MillionSongSubset/data/A/A/A/*'
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
		#print info

	h5.close()

for i in range(54):
	print i, ":", mylist[i]
	print info[i]


