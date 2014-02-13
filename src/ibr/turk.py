#!/usr/bin/python
__author__ = "Adam Vogel"

import numpy as np
import csv
import random
import string
from pybrain.utilities import n_to_one, one_to_n

def readFile(path):
  """ returns (header, lines list) """
  f = open(path)
  lines = [x for x in csv.reader(f)]
  f.close()
  header = lines[0]
  return (header, lines[1:])

# hardcoded 2d list of feature and position names
featureNames = [['hat', 'glasses', 'mustache'], \
                ['beanie', 'scarf', 'gloves'], \
                ['hat', 'scarf', 'mittens'], \
                ['cherry', 'whipped cream', 'chocolate'], \
                ['mushrooms', 'olives', 'peppers'], \
                ['sail', 'cabin', 'motor'], \
                ['ornaments', 'star', 'lights']]
positionNames = ['left', 'middle', 'right']

def lookupPosition(name):
  """ Lookup the position of a target from the name, in form [0,1,2]"""
  return positionNames.index(name)

def lookupFeature(name):
  """Lookup a feature index in [0,1,2] given a feature name like 'mustache'"""
  problemFeatures = filter(lambda features: name in features, featureNames)[0]
  return problemFeatures.index(name)

def lookupValue(name, header, line):
  """ Lookup a value from a line."""
  return line[lookupColumn(name, header)]

def lookupColumn(name, headers):
  """
  lookup a column index from a list of headers
  returns the column, zero indexed
  """
  return headers.index(name)

def parseScalesProblem(header, line):
  """Parse mechanical turk results for the "simple" condition, found in data/scales_6stimuli_3levels_no_fam_25_january_OSCA.csv
     Unfortunately the results file does not include the full description of the stimulus matrix, so we randomize the positions
     of distractors in the same way that the turk setup does. This means that repeated calls to parseScalesProblem can lead to
     slightly different problem.
  """
  problemLevel = int(lookupValue("Answer.scale_and_levels_condition", header, line)) # 0 or 1
  targetPosition = lookupPosition(string.strip(string.replace(lookupValue("Answer.target_position", header, line), '"', '')))
  targetProperty = lookupFeature(string.replace(lookupValue("Answer.target_property", header, line), '"', '')) # 0,1,2
  logicalPosition = lookupPosition(string.strip(string.replace(lookupValue("Answer.logical_position", header, line), '"', ''))) # 0,1,2
  logicalProperty = lookupFeature(string.replace(lookupValue("Answer.logical_property", header, line), '"', '')) # 0,1,2
  foilPosition = [0,1,2]
  foilPosition.remove(targetPosition)
  foilPosition.remove(logicalPosition)
  foilPosition = foilPosition[0]
  utterance = one_to_n(targetProperty, 3) 
  target = one_to_n(targetPosition, 3)
  features = np.zeros(9)
  if problemLevel == 0: # literal
    # target has both the target and the logical property
    features[3*targetPosition + targetProperty] = 1
    features[3*targetPosition + logicalProperty] = 1
    # logical has only the logical
    features[3*logicalPosition + logicalProperty] = 1
  elif problemLevel == 1:
    # target has only the target property
    features[3*targetPosition + targetProperty] = 1
    # logical has both the target and the logical property
    features[3*logicalPosition + targetProperty] = 1
    features[3*logicalPosition + logicalProperty] = 1
  else:
    print '[parseScalesProblem] Weird problemLevel:' , problemLevel
  choice = np.zeros(3)
  choiceStr = string.replace(lookupValue("Answer.choice", header, line), '"', '')
  if choiceStr == "target":
    choice = one_to_n(targetPosition, 3)
  elif choiceStr == "logical":
    choice = one_to_n(logicalPosition, 3)
  elif choiceStr == "foil":
    choice = one_to_n(foilPosition, 3)
  else:
    print "Weird Answer.choice: " , choiceStr
  return (features, utterance, target, choice, problemLevel)


def readScalesProblems(fileName, name='scales'):
  """ name 'scales' should be called with 'data/scale_plus_6stimuli_3levels_no_fam_24_january_SCAL.csv'.
      name 'scalesPlus' should be called with 'data/scale_plus_6stimuli_3levels_no_fam_24_january_SCAL.csv'
      returns an array of length #levels, with each problem as an entry.
  """
  (header, lines) = readFile(fileName)
  if name == 'scalesPlus':
    problems = [parseScalesPlusProblem(header, line) for line in lines]
    leveledProblems = [[], [], []]
  elif name == 'scales':
    problems = [parseScalesProblem(header, line) for line in lines]
    leveledProblems = [[], []]
  else:
    print "[readScalesProblems] Unknown experiment name: ", name
    return
  for problem, utterance, referent, choice, level in problems:
    leveledProblems[level].append((problem, utterance, referent, choice))  
  return leveledProblems

def parseScalesPlusProblem(header, line):
  """
  Answer.item: "snowman"
  Answer.target_property: "mittens"
  Answer.target_position: "middle"
  Answer.choice: "logical"
  (problem, utterance, referent, turkerChoice, level)
  """
  problemLevel = int(lookupValue("Answer.scale_and_levels_condition", header, line)) - 2 # in [0,1,2]
  targetPosition = lookupPosition(string.strip(string.replace(lookupValue("Answer.target_position", header, line), '"', '')))
  targetProperty = lookupFeature(string.replace(lookupValue("Answer.target_property", header, line), '"', '')) # 0,1,2

  remainingFeatureIndices = [0,1,2]
  remainingFeatureIndices.remove(targetProperty)
  random.shuffle(remainingFeatureIndices)
  logicalProperty = remainingFeatureIndices[0]
  foilProperty = remainingFeatureIndices[1]

  remainingPositionIndices = [0, 1, 2]
  remainingPositionIndices.remove(targetPosition)
  random.shuffle(remainingPositionIndices)
  logicalPosition = remainingPositionIndices[0]
  foilPosition = remainingPositionIndices[1]
  features = np.zeros(9)
  utterance = one_to_n(targetProperty, 3) 
  target = one_to_n(targetPosition, 3)

  if problemLevel == 0:
    # target has the logical and target properties
    features[3*targetPosition + targetProperty] = 1
    features[3*targetPosition + logicalProperty] = 1
    # logical face has the logical property and the foil property
    features[3*logicalPosition + logicalProperty] = 1
    features[3*logicalPosition + foilProperty] = 1
    # foil face has only the foil
    features[3*foilPosition + foilProperty] = 1
  elif problemLevel == 1:
    # the target face has only the target property, and nothing else
    features[3*targetPosition + targetProperty] = 1
    # the logical face has the target property and the logical property
    features[3*logicalPosition + targetProperty] = 1
    features[3*logicalPosition + logicalProperty] = 1
    # the foil face has the logical property and the foil property
    features[3*foilPosition + logicalProperty] = 1
    features[3*foilPosition + foilProperty] = 1
  elif problemLevel == 2:
    # the target face has the target property and the foil property
    features[3*targetPosition + targetProperty] = 1
    features[3*targetPosition + foilProperty] = 1
    # the logical face has the target property and the logical property
    features[3*logicalPosition + targetProperty] = 1
    features[3*logicalPosition + logicalProperty] = 1
    # the foil face has the foil property only
    features[3*foilPosition + foilProperty] = 1
  else:
    print "Error, unknown problemLevel: " , problemLevel
  choice = np.zeros(3)
  choiceStr = string.replace(lookupValue("Answer.choice", header, line), '"', '')
  if choiceStr == "target":
    choice = one_to_n(targetPosition, 3)
  elif choiceStr == "logical":
    choice = one_to_n(logicalPosition, 3)
  elif choiceStr == "foil":
    choice = one_to_n(foilPosition, 3)
  else:
    print "Weird Answer.choice: " , choiceStr
  return (features, utterance, target, choice, problemLevel) 

def FGToFeatures(inputString):
  """ Convert from a pragmod style matrix representation like
      '[1, 1, 0, 0, 0, 1, 0, 1, 1]'
      feature1_face1 feature1_face2 feature1_face3 .... 
      to our feature representation, which is
      [1, 0, 0,  1, 0, 1,  0, 1, 1]
      face1_feature1 face1_feature2 face1_feature3 ...
  """
  inputArray = list(inputString)
  faceFeatures = np.zeros(9)
  for faceId in range(0,3):
    # build the feature representation  
    for featureId in range(0,3):
      faceFeatures[3*faceId + featureId] = inputArray[3*featureId + faceId]
  return np.array(faceFeatures)

def loadFGProbabilities(path):
  """Load the FG model probabilities from file. Format of the file is:
    111110001 0.5 0.2 ....
    Returns a structure like map:
    (111110001,np.array([[0.5, 0.2 ..] ... ]))
    Then we do lookup by those feature definitions?
    OK, let's do our ANN representation of features.
  """
  fgFile = open(path)
  lines = [x for x in fgFile]
  fgDict = [] 
  for line in lines:
    # split by spaces.
    cols = line.strip().split(' ')   
    featureStr = cols[0]
    betStrs = cols[1:]
    featureArray = FGToFeatures(featureStr)
    betsArray = np.array(list(betStrs), dtype=float)
    betsMatrix = np.reshape(betsArray, [3, 3])
    fgDict.append((featureArray, betsMatrix))
  return fgDict

fgProbabilities = loadFGProbabilities("../../data/FG.txt")

def FGProbabilities(features, utterance):
  """Given features (in ANN style) and utterance (1-hot), return FG probabilities on each target face."""
  matchingProbabilities = [i[1] for i in fgProbabilities if (i[0] == features).all()]
  probs = matchingProbabilities[0]
  return probs[n_to_one(utterance)]

IBR_L0 = loadFGProbabilities("../../data/IBR-L0.txt")
IBR_LS = loadFGProbabilities("../../data/IBR-LS.txt")
IBR_LSL = loadFGProbabilities("../../data/IBR-LSL.txt")
IBR_LSLS = loadFGProbabilities("../../data/IBR-LSLS.txt")
IBR_LSLSL = loadFGProbabilities("../../data/IBR-LSLSL.txt")
IBR_LSLSLS = loadFGProbabilities("../../data/IBR-LSLSLS.txt")
IBR_LSLSLSLS = loadFGProbabilities("../../data/IBR-LSLSLSLS.txt")
IBR_FG = loadFGProbabilities("../../data/IBR-FG.txt")

def IBRProbabilities(features, utterance, name="L0"):
  if name == "L0":
    return IBR(features, utterance, IBR_L0)
  elif name == "LS":
    return IBR(features, utterance, IBR_LS)
  elif name == "LSL":
    return IBR(features, utterance, IBR_LSL)
  elif name == "LSLS":
    return IBR(features, utterance, IBR_LSLS)
  elif name == "LSLSL":
    return IBR(features, utterance, IBR_LSLSL)
  elif name == "LSLSLS":
    return IBR(features, utterance, IBR_LSLSLS)
  elif name == "LSLSLSLS":
    return IBR(features, utterance, IBR_LSLSLSLS)
  elif name == "FG":
    return IBR(features, utterance, IBR_FG)
  else:
    print "[IBRProbabilities] unknown IBR model name: " , name

def IBR(features, utterance, modelProbabilities):
  matchingProbabilities = [i[1] for i in modelProbabilities if (i[0] == features).all()]
  probs = matchingProbabilities[0]
  return probs[n_to_one(utterance)]
