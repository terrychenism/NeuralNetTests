import numpy as np
from scipy.io import loadmat
from glob import glob
from datetime import datetime

loc_data_dir_train = "data/train/*.mat" 
loc_data_dir_test = 'data/test/*.mat'

#will be created
loc_vw_train = "face.train.vw"
loc_vw_test = "face.test.vw"

def create_features(XX, tmin, tmax, sfreq, tmin_original=-0.5):
	"""Creation of the feature space:
	- restricting the time window of MEG data to [tmin, tmax]sec.
	- Concatenating the 306 timeseries of each trial in one long
	  vector.
	- Normalizing each feature independently (z-scoring).
	"""
	print "Applying the desired time window."
	beginning = np.round((tmin - tmin_original) * sfreq).astype(np.int)
	end = np.round((tmax - tmin_original) * sfreq).astype(np.int)
	XX = XX[:, :, beginning:end].copy()
	
	print "2D Reshaping: concatenating all 306 timeseries."
	XX = XX.reshape(XX.shape[0], XX.shape[1] * XX.shape[2])

	print "Features Normalization."
	XX -= XX.mean(0)
	XX = np.nan_to_num(XX / XX.std(0))

	return XX

def to_vw(glob_data, loc_vw_file, train = True):	
	"""Loads the datasets and creates Vowpal Wabbit-formatted files"""
	lines_wrote = 0
	start = datetime.now()
	with open(loc_vw_file, "wb") as outfile:
		for file_nr, glob_file in enumerate( glob(glob_data) ):
			print "\n\nLoading:", glob_file, file_nr+1, "/", len(glob(glob_data))
			data = loadmat(glob_file, squeeze_me=True)
			XX = data['X']
			
			sfreq = data['sfreq']
			tmin_original = data['tmin']
			print "\nDataset summary:"
			print "XX:", XX.shape
			if train:
				yy = data['y']
				print "yy:", yy.shape
			print "sfreq:", sfreq

			# We throw away all the MEG data outside the first 0.5sec from when
			# the visual stimulus start:
			tmin = 0.0
			tmax = 0.500
			print "\nRestricting MEG data to the interval [%s, %s] sec." % (tmin, tmax)

			XX = create_features(XX, tmin, tmax, sfreq)

			print "\nAdding to Vowpal Wabbit formatted file:", loc_vw_file
			print "\n#sample\t#total\t#wrote\ttime spend" 
			for trial_nr, X in enumerate(XX):
				outline = ""
				if train:
					if yy[trial_nr] == 1:
						label = 1
					else:
						label = -1 #change label from 0 to -1 for binary
					outline += str(label) + " '" + str(lines_wrote) + " |f"
				else:
					label = 1 #dummy label for test set
					id = 17000 + (file_nr*1000) + trial_nr
					outline += str(label) + " '" + str(id) + " |f"
				for feature_nr, val in enumerate(X):
					outline += " " + str(feature_nr) + ":" + str(val)
				outfile.write( outline + "\n" )
				lines_wrote += 1
				if trial_nr % 100 == 0:
					print "%s\t%s\t%s\t%s" % (trial_nr, XX.shape[0], lines_wrote, datetime.now()-start)

if __name__ == '__main__':
	start = datetime.now()
	to_vw(loc_data_dir_train, loc_vw_train, train = True)
	to_vw(loc_data_dir_test, loc_vw_test, train = False)
	print "\nTotal script running time:", datetime.now()-start
