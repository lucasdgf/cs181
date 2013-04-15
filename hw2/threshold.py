"""
Coded with love by Lucas Freitas
Harvard University Class of 2015 
"""

import re
import sys
from operator import itemgetter

def tokenize (filename):
	"""
	Takes a text file name and returns a Python list    
	a tokenized version of the text of the   

	Args:
		filename: name of the text file to be tokenized
	Returns:
		List of strings
	Raises:
		In case of IOError, prints error and exits
	"""

	try:
		file = open(filename, 'r')
	except IOError:
		print "IOError: could not open", filename
		sys.exit()

	perfs = []

	for line in file.readlines():
		list = re.findall(r'[\w\.]+', line)
		perfs.append(float(list[3]))

	return perfs

def main ():
	perfs = tokenize ("15.txt")

	# list of last 5 validation set performances
	perf_list = []

	# Loop through the specified number of training epochs.
	for i in range(100):
		perf_validate = perfs[i]
      
		# check if threshold if reache0d
		if i >= 10:
			if perf_validate - (sum(perf_list))/10 < 0.001:
				print i + 1, perf_validate
				sys.exit()

		# update performance list
		if len(perf_list) >= 10:
			perf_list.pop(0)
			perf_list.append(perf_validate)
		else:
			perf_list.append(perf_validate)

if __name__ == "__main__":
	main()
