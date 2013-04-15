from neural_net import NeuralNetwork, NetworkFramework
from neural_net import Node, Target, Input
import random


# <--- Problem 3, Question 1 --->

def FeedForward(network, input):
  """
  Arguments:
  ---------
  network : a NeuralNetwork instance
  input   : an Input instance

  Returns:
  --------
  Nothing

  Description:
  -----------
  This function propagates the inputs through the network. That is,
  it modifies the *raw_value* and *transformed_value* attributes of the
  nodes in the network, starting from the input nodes.

  Notes:
  -----
  The *input* arguments is an instance of Input, and contains just one
  attribute, *values*, which is a list of pixel values. The list is the
  same length as the number of input nodes in the network.

  i.e: len(input.values) == len(network.inputs)

  This is a distributed input encoding (see lecture notes 7 for more
  informations on encoding)

  In particular, you should initialize the input nodes using these input
  values:

  network.inputs[i].raw_value = input[i]
  """
  network.CheckComplete()

  # 0) Check if list of values has same length as number of input nodes
  assert len(input.values) == len(network.inputs)

  # 1) Assign input values to input nodes
  for i in range(len(input.values)):
    network.inputs[i].raw_value = input.values[i]
    network.inputs[i].transformed_value = input.values[i]

  # 2) Propagates to hidden layer
  for node in network.hidden_nodes:
    node.raw_value = NeuralNetwork.ComputeRawValue(node)
    # use sigmoid to get transformed value
    node.transformed_value = NeuralNetwork.Sigmoid(node.raw_value)

  # 3) Propagates to the output layer
  for node in network.outputs:
    node.raw_value = NeuralNetwork.ComputeRawValue(node)
    node.transformed_value = NeuralNetwork.Sigmoid(node.raw_value)

#< --- Problem 3, Question 2

def Backprop(network, input, target, learning_rate):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  input         : an Input instance
  target        : a target instance
  learning_rate : the learning rate (a float)

  Returns:
  -------
  Nothing

  Description:
  -----------
  The function first propagates the inputs through the network
  using the Feedforward function, then backtracks and update the
  weights.

  Notes:
  ------
  The remarks made for *FeedForward* hold here too.

  The *target* argument is an instance of the class *Target* and
  has one attribute, *values*, which has the same length as the
  number of output nodes in the network.

  i.e: len(target.values) == len(network.outputs)

  In the distributed output encoding scenario, the target.values
  list has 10 elements.

  When computing the error of the output node, you should consider
  that for each output node, the target (that is, the true output)
  is target[i], and the predicted output is network.outputs[i].transformed_value.
  In particular, the error should be a function of:

  target[i] - network.outputs[i].transformed_value
  
  """
  network.CheckComplete()

  # 0) Check if list of values has same length as number of output nodes
  assert len(target.values) == len(network.outputs)

  # 1) We first propagate the input through the network (process children)
  FeedForward(network, input)

  # 2) Then we compute the errors and update the weights starting with the last layer
  # 2.1) Compute the errors, storing them in a dictionary
  
  errors = {}

  for m in range(len(network.outputs)):
    epsilon = target.values[m] - network.outputs[m].transformed_value
    delta = NeuralNetwork.SigmoidPrime(network.outputs[m].raw_value) * epsilon
    # index with object, since we will also calculate errors for hidden units
    errors[network.outputs[m]] = delta

  # 2.2) Update the weights starting with the last layer

  length = len(network.outputs) - 1

  for i in range(length + 1):
    j = length - i
    node = network.outputs[j]
    for input in node.inputs:
      input.forward_weights[j].value += learning_rate * input.transformed_value * errors[node]

  # 3) We now propagate the errors to the hidden layer, and update the weights there too
  # 3.1) Compute the errors, storing them in the dictionary we created in 2.1

  length = len(network.hidden_nodes) - 1

  for i in range(length + 1):
    m = length - i
    node = network.hidden_nodes[m]

    epsilon = 0.
    for j in range(len(node.forward_neighbors)):
      epsilon += node.forward_weights[j].value * errors[node.forward_neighbors[j]]

    delta = NeuralNetwork.SigmoidPrime(node.raw_value) * epsilon
    errors[node] = delta

  # 3.2) Update the weights starting with the last layer

  length = len(network.hidden_nodes) - 1

  for i in range(length + 1):
    j = length - i
    node = network.hidden_nodes[j]
    for input in node.inputs:
      input.forward_weights[j].value += learning_rate * input.transformed_value * errors[node]

# <--- Problem 3, Question 3 --->

def Train(network, inputs, targets, learning_rate, epochs):
  """
  Arguments:
  ---------
  network       : a NeuralNetwork instance
  inputs        : a list of Input instances
  targets       : a list of Target instances
  learning_rate : a learning_rate (a float)
  epochs        : a number of epochs (an integer)

  Returns:
  -------
  Nothing

  Description:
  -----------
  This function should train the network for a given number of epochs. That is,
  run the *Backprop* over the training set *epochs*-times
  """
  network.CheckComplete()

  # check if inputs and targets lists have the same length
  assert len(inputs) == len(targets)

  # run *epoches*-times
  for i in range(epochs):
    for j in range(len(inputs)):
      target = Target()
      target.values = targets[j]
      Backprop(network, inputs[j], target, learning_rate)

# <--- Problem 3, Question 4 --->

class EncodedNetworkFramework(NetworkFramework):
  def __init__(self):
    """
    Initializatio.
    YOU DO NOT NEED TO MODIFY THIS __init__ method
    """
    super(EncodedNetworkFramework, self).__init__() # < Don't remove this line >
    
  # <--- Fill in the methods below --->

  def EncodeLabel(self, label):
    """
    Arguments:
    ---------
    label: a number between 0 and 9

    Returns:
    ---------
    a list of length 10 representing the distributed
    encoding of the output.

    Description:
    -----------
    Computes the distributed encoding of a given label.

    Example:
    -------
    0 => [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    3 => [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Notes:
    ----
    Make sure that the elements of the encoding are floats.
    
    """
    # verify that label is between 0 and 9
    assert(0 <= label and label <= 9)

    # Replace line below by content of function
    return [1.0 if x == label else 0.0 for x in range(10)]

  def GetNetworkLabel(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    the 'best matching' label corresponding to the current output encoding

    Description:
    -----------
    The function looks for the transformed_value of each output, then decides 
    which label to attribute to this list of outputs. The idea is to 'line up'
    the outputs, and consider that the label is the index of the output with the
    highest *transformed_value* attribute

    Example:
    -------

    # Imagine that we have:
    map(lambda node: node.transformed_value, self.network.outputs) => [0.2, 0.1, 0.01, 0.7, 0.23, 0.31, 0, 0, 0, 0.1, 0]

    # Then the returned value (i.e, the label) should be the index of the item 0.7,
    # which is 3
    
    """
    # Replace line below by content of function
    outputs = [node.transformed_value for node in self.network.outputs]
    return outputs.index(max(outputs))

  def Convert(self, image):
    """
    Arguments:
    ---------
    image: an Image instance

    Returns:
    -------
    an instance of Input

    Description:
    -----------
    The *image* arguments has 2 attributes: *label* which indicates
    the digit represented by the image, and *pixels* a matrix 14 x 14
    represented by a list (first list is the first row, second list the
    second row, ... ), containing numbers whose values are comprised
    between 0 and 256.0. The function transforms this into a unique list
    of 14 x 14 items, with normalized values (that is, the maximum possible
    value should be 1).
    
    """
    # Replace line below by content of function
    values = []
    for i in range(len(image.pixels)):
      for j in range(len(image.pixels[i])):
        values.append(image.pixels[i][j] / 256.0)
    input = Input()
    input.values = values
    return input

  def InitializeWeights(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes the weights with random values between [-0.01, 0.01].

    Hint:
    -----
    Consider the *random* module. You may use the the *weights* attribute
    of self.network.
    
    """
    # replace line below by content of function
    for weight in self.network.weights:
      weight.value = random.uniform(-0.01, 0.01)

#<--- Problem 3, Question 6 --->

class SimpleNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a simple network, with 196 input nodes,
    10 output nodes, and NO hidden nodes. Each input node
    should be connected to every output node.
    """
    super(SimpleNetwork, self).__init__() # < Don't remove this line >
    
    # 1) Adds an input node for each pixel.
    for i in range(196):
      self.network.AddNode(Node(), NeuralNetwork.INPUT)

    # 2) Add an output node for each possible digit label.
    for i in range(10):
      node = Node()
      # connect each input node to the soon-to-be output node
      for input in self.network.inputs:
        node.AddInput(input, None, self.network)
      # add the output node to the network
      self.network.AddNode(node, NeuralNetwork.OUTPUT)

#<---- Problem 3, Question 7 --->

class HiddenNetwork(EncodedNetworkFramework):
  def __init__(self, number_of_hidden_nodes=30):
    """
    Arguments:
    ---------
    number_of_hidden_nodes : the number of hidden nodes to create (an integer)

    Returns:
    -------
    Nothing

    Description:
    -----------
    Initializes a network with a hidden layer. The network
    should have 196 input nodes, the specified number of
    hidden nodes, and 10 output nodes. The network should be,
    again, fully connected. That is, each input node is connected
    to every hidden node, and each hidden_node is connected to
    every output node.
    """
    super(HiddenNetwork, self).__init__() # < Don't remove this line >

    # 1) Adds an input node for each pixel.
    for i in range(196):
      self.network.AddNode(Node(), NeuralNetwork.INPUT)

    # 2) Adds the hidden layer
    for i in range(number_of_hidden_nodes):
      node = Node()
      # connect each input node to the soon-to-be hidden node
      for input in self.network.inputs:
        node.AddInput(input, None, self.network)
      # add the hidden node to the network
      self.network.AddNode(node, NeuralNetwork.HIDDEN)

    # 3) Adds an output node for each possible digit label.
    for i in range(10):
      node = Node()
      # connect each hidden node to the soon-to-be output node
      for hidden in self.network.hidden_nodes:
        node.AddInput(hidden, None, self.network)
      # add the output node to the network
      self.network.AddNode(node, NeuralNetwork.OUTPUT)
    
#<--- Problem 3, Question 8 ---> 

class CustomNetwork(EncodedNetworkFramework):
  def __init__(self):
    """
    Arguments:
    ---------
    Nothing

    Returns:
    --------
    Nothing

    Description:
    -----------
    Initializes a simple network, with 196 input nodes,
    10 output nodes, and NO hidden nodes. Each input node
    should be connected to every output node. The big
    difference of this network, however, is that if considers
    pixels as bits - if they are mostly dark (values 128 or
    higher), we convert them to a 1, and otherwise, to a 0.
    We then analyze how the performance of the method compares
    to the one for Simple or Hidden networks with pixels in a
    scale from 0 to 1.
    """
    super(CustomNetwork, self).__init__() # <Don't remove this line>

    # 1) Adds an input node for each pixel.
    for i in range(196):
      self.network.AddNode(Node(), NeuralNetwork.INPUT)

    # 2) Add an output node for each possible digit label.
    for i in range(10):
      node = Node()
      # connect each input node to the soon-to-be output node
      for input in self.network.inputs:
        node.AddInput(input, None, self.network)
      # add the output node to the network
      self.network.AddNode(node, NeuralNetwork.OUTPUT)
  
    def Convert(self, image):
      """
      Arguments:
      ---------
      image: an Image instance

      Returns:
      -------
      an instance of Input

      Description:
      -----------
      The *image* arguments has 2 attributes: *label* which indicates
      the digit represented by the image, and *pixels* a matrix 14 x 14
      represented by a list (first list is the first row, second list the
      second row, ... ), containing numbers whose values are comprised
      between 0 and 256.0. The function transforms this into a unique list
      of 14 x 14 items, with binary values, i.e., 1 if the pixel is mostly
      dark, or 0 if it's mostly light.
      
      """
      # Replace line below by content of function
      values = []
      for i in range(len(image.pixels)):
        for j in range(len(image.pixels[i])):
          if image.pixels[i][j] >= 128:
            pixel = 1
          else:
            pixel = 0
          values.append(pixel)
      input = Input()
      input.values = values
      return input
