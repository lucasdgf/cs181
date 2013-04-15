import math
import random

# normal PDF
def N(x, mean, var):
    num = math.exp(-(float(x) - float(mean)) ** 2 / (2 * var))
    denom = (2 * math.pi * var) ** .5
    return num / denom

# mixture of gaussians
def mixture(x):
	return 0.2 * N(x, 1, 25) + 0.3 * N(x, -2, 1) + 0.5 * N(x, 3, 4)

# rejection sampling
def sample(filename, k, c, mu, sigma):
	file = open(filename, "w")
	i = 0
	random.seed()
	while i < k:
		x = c * random.gauss(mu, sigma)
		if N(x, mu, sigma * sigma) <= mixture(x):
			file.write(str(x) + ",\n")
			i += 1
	file.close()

sample("data-rejection.txt", 500, 2, 0, 5)