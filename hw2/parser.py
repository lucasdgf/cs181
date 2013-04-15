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

	for line in file.readlines():
		list = re.findall(r'[\w\.]+', line)
		print list[4] + ","

def main ():
	tokenize ("30.txt")

if __name__ == "__main__":
	main()
