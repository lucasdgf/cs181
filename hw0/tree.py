class tree():
  """Class for a tree"""

  def __init__(self, data):
    self.data = data
    self.children = []
    pass

  def addChild (self, child):
    self.children.append(child)

  def removeChild (self, child):
    self.children.remove(child)

n = tree(5)
p = tree(6)
q = tree(7)

n.addChild(p)
n.addChild(q)
n.removeChild(q)

for c in n.children:
  print c.data