# Import the math module, so that we have access to the functions sqrt and pow
import math

def factorial(x):
  """Return x!, assuming that x is a non-negative integer."""
  if x == 0:
    return 1
  else:
    return x * factorial(x - 1)

def sumFile(filename):
  """Each line of filename contains a float.  Return the sum of all lines in the
  file."""
  file = open(filename, 'r')
  sum = 0.
  for line in file.readlines():
    sum += float(line)
  return sum


# This is the syntax for a Python class.
class Point():
  """Class that encapsulates a single point in the x-y plane."""

  # This is the constructor for the class.  By convention, the first argument to
  # any method of the class is self, referring to the variable itself.  This is
  # similar to the "this" variable in other programming languages.
  def __init__(self, x_coord, y_coord):
    self.x_coord = x_coord
    self.y_coord = y_coord
    pass

  def distanceFromOrigin(self):
    return pow(pow(self.x_coord, 2) + pow(self.y_coord, 2), 0.5)
