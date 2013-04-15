# main.py
# -------
# Lucas Freitas '15 and Angela Li '14

import matplotlib.pyplot as plt
from dtree import *
from pylab import *
import sys, random, copy, math

class Globals:
    noisyFlag = False
    pruneFlag = False
    valSetSize = 0
    dataset = None

##Classify
#---------

def classify(decisionTree, example):
    return decisionTree.predict(example)

def classify_weights(decisionTrees, example):
    """Uses a weighted set of decision trees to classify an instance."""
    votes = {}
    for dt in decisionTrees:
        v = dt.predict(example)
        votes[v] = (votes[v] + dt.weight if v in votes else dt.weight)
    return max(votes, key=votes.get)

##Learn
#-------
def learn(dataset):
    learner = DecisionTreeLearner()
    learner.train(dataset)
    return learner.dt

def dumb_learn(dataset):
    learner = MajorityLearner()
    learner.train(dataset)
    return learner.dt

def adaboost_wrapper(dataset, rounds):
    """Returns a voting hypothesis consisting of a set of weighted decision trees."""
    # normalize data weights
    for e in dataset.examples:
        e.weight = float(1) / len(dataset.examples)

    committee = []

    # boosting rounds
    for i in range(rounds):
        # create hypothesis based on weighed data
        hypothesis = learn(dataset)

        # calculate hypothesis error
        h_error = sum([e.weight * (hypothesis.predict(e) != e.attrs[-1]) for e in dataset.examples])
        if not h_error:
            return [hypothesis]
        
        # assign weight to hypothesis
        h_weight = 0.5 * math.log((1 - h_error) / h_error)
        hypothesis.weight = h_weight
        committee.append(hypothesis)
        
        # re-weight and re-normalize data
        for e in dataset.examples:
            sign = 1 if hypothesis.predict(e) != e.attrs[-1] else -1
            e.weight *= math.exp(h_weight * sign)

        weight_sum = sum([e.weight for e in dataset.examples])
        for e in dataset.examples:
            e.weight = e.weight / weight_sum

    return committee

# main
# ----
# The main program loop
# You should modify this function to run your experiments

def parseArgs(args):
  """Parses arguments vector, looking for switches of the form -key {optional value}.
  For example:
    parseArgs([ 'main.py', '-n', '-p', 5 ]) = { '-n':True, '-p':5 }"""
  args_map = {}
  curkey = None
  for i in xrange(1, len(args)):
    if args[i][0] == '-':
      args_map[args[i]] = True
      curkey = args[i]
    else:
      assert curkey
      args_map[curkey] = args[i]
      curkey = None
  return args_map

def validateInput(args):
    args_map = parseArgs(args)
    valSetSize = 0
    noisyFlag = False
    pruneFlag = False
    boostRounds = -1
    maxDepth = -1
    if '-n' in args_map:
      noisyFlag = True
    if '-p' in args_map:
      pruneFlag = True
      valSetSize = int(args_map['-p'])
    if '-d' in args_map:
      maxDepth = int(args_map['-d'])
    if '-b' in args_map:
      boostRounds = int(args_map['-b'])
    return [noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds]

def plot_pruning(dataset, noisyFlag):
    length = (len(dataset.examples) / 2) / 10
    ys = []
    zs = []
    max_test = 0.
    test_index = 0
    max_train = 0.
    train_index = 0

    plt.clf()

    for n in range(1, 81):
      train_scores = []
      test_scores = []
      for i in range(10):

        # Divide the dataset into train and test sets
        train = copy.deepcopy(dataset)
        train.examples = train.examples[length * i : length * (i + 9)]
        test = copy.deepcopy(dataset)
        test.examples = test.examples[length * (i + 9) : 10 * (i + 10)]
        
        # Devide the train set into train and validation sets
        validation = copy.deepcopy(train)
        train.examples = train.examples[n:]
        validation.examples = validation.examples[:n]

        # Create a tree
        learned_tree = learn(train)

        # Prune it
        learned_tree.prune(validation)

        # Find out the performances for train and test sets and append
        # score to score list, so we can average scores after 10 runs
        test_scores.append(learned_tree.score(test))
        train_scores.append(learned_tree.score(train))

      test_perf = float(sum(test_scores)) / len(test_scores)
      train_perf = float(sum(train_scores)) / len(train_scores)
      ys.append(test_perf)
      zs.append(train_perf)

      if train_perf > max_train:
          max_train = train_perf
          train_index = n
      if test_perf > max_test:
          max_test = test_perf
          test_index = n

      print "Performance{0}: {1}".format((" on noisy data" if noisyFlag else " on non-noisy data"), test_perf)

    print "Maximum train performance at {0} with value {1}".format(train_index, max_train)
    print "Maximum test performance at {0} with value {1}".format(test_index, max_test)

    xs = range(1, 81)
    p1, = plt.plot(xs, ys, color='b')
    p2, = plt.plot(xs, zs, color='r')
    if noisyFlag:
      plt.title('Noisy Pruning')
    else:
      plt.title('Non-noisy Pruning')
    plt.xlabel('Validation Set Size')
    plt.ylabel('Prediction Accuracy')
    plt.axis([1, 80, 0.5, 1])
    plt.legend((p1,p2), ('Test Data','Train Data'), 'lower right')
  
    # save the figure to a file
    if noisyFlag:
      savefig('noisy.pdf')
    else:
      savefig('non-noisy.pdf')
    plt.show() # show the figure

def committee_score(committee, dataset):
        total, correct = 0, 0
        for e in dataset.examples:
            total += 1
            if classify_weights(committee, e) == e.attrs[-1]:
                correct += 1
        return float(correct) / total if total else 0

def plot_boosting(dataset, noisyFlag):
    length = (len(dataset.examples) / 2) / 10
    ys = []
    zs = []
    max_test = 0.
    test_index = 0
    max_train = 0.
    train_index = 0

    plt.clf()

    for n in range(1, 31):
      dataset.num_rounds = n
      train_scores = []
      test_scores = []
      for i in range(10):

        # Divide the dataset into train and test sets
        train = copy.deepcopy(dataset)
        train.examples = train.examples[length * i : length * (i + 9)]
        test = copy.deepcopy(dataset)
        test.examples = test.examples[length * (i + 9) : 10 * (i + 10)]

        # Create committee
        learned_committee = adaboost_wrapper(train, dataset.num_rounds)
        
        
        # Find out the performances for train and test sets and append
        # score to score list, so we can average scores after 10 runstest_score = committee_score(learned_committee, test)
        test_score = committee_score(learned_committee, test)
        test_scores.append(test_score)
        train_score = committee_score(learned_committee, train)
        train_scores.append(train_score)

      test_perf = float(sum(test_scores)) / len(test_scores)
      train_perf = float(sum(train_scores)) / len(train_scores)
      ys.append(test_perf)
      zs.append(train_perf)

      if train_perf > max_train:
          max_train = train_perf
          train_index = n
      if test_perf > max_test:
          max_test = test_perf
          test_index = n

    print "Maximum train performance at {0} with value {1}".format(train_index, max_train)
    print "Maximum test performance at {0} with value {1}".format(test_index, max_test)

    xs = range(1, 31)
    p1, = plt.plot(xs, ys, color='b')
    p2, = plt.plot(xs, zs, color='r')
    if noisyFlag:
      plt.title('Noisy Boosting for Depth 1')
    else:
      plt.title('Non-noisy Boosting for Depth 1')
    plt.xlabel('Boosting Rounds')
    plt.ylabel('Prediction Accuracy')
    plt.axis([1, 30, 0.8, 1])
    plt.legend((p1,p2), ('Test Data','Train Data'), 'lower right')
  
    # save the figure to a file
    if noisyFlag:
      savefig('noisy-boosting-depth-1.pdf')
    else:
      savefig('non-noisy-boosting-depth-1.pdf')
    plt.show() # show the figure

def main():
    arguments = validateInput(sys.argv)
    noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds = arguments
    print noisyFlag, pruneFlag, valSetSize, maxDepth, boostRounds

    # Read in the data file
    
    if noisyFlag:
        f = open("noisy.csv")
    else:
        f = open("data.csv")

    data = parse_csv(f.read(), " ")
    dataset = DataSet(data)
    
    # Copy the dataset so we have two copies of it
    examples = dataset.examples[:]
 
    dataset.examples.extend(examples)
    dataset.max_depth = maxDepth
    if boostRounds != -1:
      dataset.use_boosting = True
      dataset.num_rounds = boostRounds
      # Uncomment to plot boosting graph
      #plot_boosting(dataset, noisyFlag)
    
    # Uncomment to plot pruning graph
    # else:
      # plot_pruning(dataset, noisyFlag)

    length = (len(dataset.examples) / 2) / 10
    test_scores = []
    train_scores = []

    for i in range(0, 10):
        train = copy.deepcopy(dataset)
        train.examples = train.examples[length * i : length * (i + 9)]
        test = copy.deepcopy(dataset)
        test.examples = test.examples[length * (i + 9) : 10 * (i + 10)]

        if (boostRounds >= 0):
          learned_committee = adaboost_wrapper(train, dataset.num_rounds)
          test_score = committee_score(learned_committee, test)
          test_scores.append(test_score)
          train_score = committee_score(learned_committee, train)
          train_scores.append(train_score)
        else:
          learned_tree = learn(train)
          test_score = learned_tree.score(test)
          test_scores.append(test_score)
          train_score = learned_tree.score(train)
          train_scores.append(train_score)

        print train_score, test_score

    test_perf = float(sum(test_scores)) / len(test_scores)
    train_perf = float(sum(train_scores)) / len(train_scores)

    print "Training average: {0}, testing average: {1}".format(train_perf, test_perf)

main()

    
