import random

def mixture(filename, k):
	file = open(filename, "w")
	for i in range(k):
		n = random.randrange(0,10)
		if n < 2:
			file.write(str(random.gauss(1,5)) + ",\n")
		elif n < 5:
			file.write(str(random.gauss(-2,1)) + ",\n")
		else:
			file.write(str(random.gauss(3,2)) + ",\n")
	file.close()

random.seed()
filename = "data.txt"
k = 500
mixture(filename, k)