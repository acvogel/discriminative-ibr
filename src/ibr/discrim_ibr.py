#!/usr/bin/python
__author__ = "Adam Vogel"
__email__ = "acvogel@gmail.com"

import turk
import numpy as np
import random
import itertools
import scipy
import pickle
import csv
import sys
import matplotlib
import matplotlib.pyplot as plt
from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import n_to_one, one_to_n
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection
import scikits.bootstrap as boot

numFeatures = 12 # 9 face features, 3 features for utterance (listener) or target (speaker)
numFaces = 3
numProperties = 3
colors = [matplotlib.colors.hex2color(color) for color in ['#83ff0d', '#1700ff', '#ff460d']]

def literalListener(features, utterance):
  """ Given an input feature, and a 1-hot utterance, 
  pick randomly from faces for which that is true.
  Assumption: features are ordered like 
  [face_0_property_0 face_0_property_1 ... ]
  input is a property.
  utterance is a 1-hot vector, 1 for each property
  returns a 1-hot vector of length numFaces, 1 for chosen face
  """
  propertyIdx = utterance.nonzero()[0]
  faceTruthValues = np.zeros(numFaces)
  for faceId in range(numFaces):
    # slice into relevant face feature
    faceFeatures = features[faceId * numProperties : (faceId + 1) * numProperties]
    faceTruthValues[faceId] = faceFeatures[propertyIdx]
  trueFaceIndices = np.flatnonzero(faceTruthValues)
  # an utterance which iz true of nothing iz true of everything, lebowski!
  if trueFaceIndices.size == 0 :
    trueFaceIndices = np.zeros(numFaces)
    trueFaceIndices.fill(1)
  choice = random.choice(trueFaceIndices)
  answerArray = np.zeros(numFaces)
  answerArray[choice] = 1
  return answerArray

def literalSpeaker(features, target):
  """ Given an input feature and a 1-hot target, 
  pick a random true utterance which applies to the target.
  Returns a 1-hot representation of that utterance.
  """
  # figure out the target, slice out the features, pick a random 1, return vector
  targetIdx = target.nonzero()[0]
  targetFeatures = features[targetIdx*numProperties : (targetIdx + 1) * numProperties]
  # now, pick a random one which is nonzero
  trueFeatureIndices = np.flatnonzero(targetFeatures)
  # if a face has no features, pick a random one.
  if trueFeatureIndices.size == 0:
    trueFeatureIndices = np.zeros(numProperties)
    trueFeatureIndices.fill(1)
  choice = random.choice(trueFeatureIndices) 
  answerArray = np.zeros(numProperties)
  answerArray[choice] = 1
  return answerArray

def targets():
  return [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]

def utterances():
  return [np.array([1,0,0]), np.array([0,1,0]), np.array([0,0,1])]

def goldListenerTrainingExamplesFromInstances(instances):
  """Construct a ClassificationDataSet for a set of instances,
     which are of the form (features, utterance, target).
     problem[0] = concat(features, utterance)
     problem[1] = target (label)
     Here, utterance uses the "gold" label, not a specific speaker.
  """
  dataset = ClassificationDataSet(numFeatures, 3, nb_classes=3, class_labels=['face0', 'face1', 'face2'])
  for (features, utterance, target) in instances:
    dataset.addSample(np.concatenate([features, utterance]), target)
  return dataset


def listenerTrainingExamplesFromInstances(problems, speaker):
  """Construct a ClassificationDataSet for a set of problems.
     "problems" is a dataset with 1 row per faces problem, of the form (features, utterance, target)
     Here utterance will be ignored, instead using the speaker utterance
  """
  dataset = ClassificationDataSet(numFeatures, 3, nb_classes=3, class_labels=['face0', 'face1', 'face2'])
  for problem in problems:
    features = problem[0]
    target = problem[2]
    utterance = speaker(features, target)
    dataset.addSample(np.concatenate([features, utterance]), target)
  return dataset


def listenerANNToFunction(listenerANN):
  """ Convert a listener ANN to one that classifies incoming data.
    A listener is a function which takes 2 arguments: features, and utterance, and returns 
    a 1 hot representation of the target
    literalListener(features, utterance)
  """
  def listener(features, utterance):
    activations = listenerANN.activate(np.concatenate([features, utterance]))
    # now, convert activations to 1 hot 
    activationMax = activations.argmax()
    target = np.zeros(3)
    target[activationMax] = 1
    return target
  return listener

def speakerANNToFunction(speakerANN):
  """ Convert a speaker ANN to one that classifies incoming data.
      A speaker is a function which takes 2 arguments: features, and target,
      and returns a 1-hot representation of the utterance
      literalSpeaker(features, target):
  """
  def speaker(features, target):
    activations = speakerANN.activate(np.concatenate([features, target]))
    activationMax = activations.argmax()
    utterance = np.zeros(3)
    utterance[activationMax] = 1
    return utterance
  return speaker

def speakerTrainingExamples(problems, listener):
  """given problems, and a listener, iterate over targets, produce utterance"""
  dataset = ClassificationDataSet(numFeatures, 3, nb_classes=3, class_labels=['glasses', 'hat', 'moustache'])
  for features in problems:
    for target in targets():
      for i in range(3): # add multiple instances to deal with randomness
        bestUtterances = [utterance for utterance in utterances() if (listener(features, utterance) == target).all()]
        if not bestUtterances:
          bestUtterances = utterances()
        bestUtterance = random.choice(bestUtterances)
        dataset.addSample(np.concatenate([features, target]), bestUtterance)
  return dataset

def speakerTrainingExamplesFromInstances(problems, listener):
  dataset = ClassificationDataSet(numFeatures, 3, nb_classes=3, class_labels=['glasses', 'hat', 'moustache'])
  for problem in problems:
    features = problem[0]
    target = problem[2]
    bestUtterances = [utterance for utterance in utterances() if (listener(features, utterance) == target).all()]
    if not bestUtterances:
      bestUtterances = utterances()
    bestUtterance = random.choice(bestUtterances)
    dataset.addSample(np.concatenate([features, target]), bestUtterance)
  return dataset

def initNetwork(layerSizes):
  """Builds a network with hidden layer sizes specified in the argument. 
     Output is a linear layer of 3 outputs, input layer has numFeatures.
     Intermediate layers are all sigmoids. 
  """
  fnn = FeedForwardNetwork()
  inLayer = LinearLayer(numFeatures)
  fnn.addInputModule(inLayer)
  outLayer = LinearLayer(3)
  fnn.addOutputModule(outLayer)
  hiddenLayers = [SigmoidLayer(size) for size in layerSizes]
  for hiddenLayer in hiddenLayers:
    fnn.addModule(hiddenLayer)
  allLayers = list(itertools.chain.from_iterable([[inLayer], hiddenLayers, [outLayer]]))
  # iterate over successive pairs of layers
  for layer1, layer2 in zip(allLayers[:-1], allLayers[1:]): 
    fnn.addConnection(FullConnection(layer1, layer2))
  fnn.sortModules()
  return fnn

def trainAgent(trndata, iterations, hiddenNodes=[24]):
  """Train an agent from a partner-derived dataset. Returns the ANN and the trainer"""
  fnn = initNetwork(hiddenNodes) # simple network structure
  trainer = BackpropTrainer(fnn, dataset=trndata, verbose=False)
  trainer.trainUntilConvergence(verbose=False, maxEpochs=iterations)
  return (fnn, trainer)

def loadFacesFeatures(featurePath):
 return np.loadtxt(featurePath, delimiter=",")

def trainSpeaker(problems, listener, hiddenNodes=[24]):
  """ here listener is assumed to be in functional form
      this returns ANN form.   
  """
  #dataset = speakerTrainingExamples(problems, listener)
  dataset = speakerTrainingExamplesFromInstances(problems, listener)
  (speaker, trainer) = trainAgent(dataset, 300, hiddenNodes)
  return speaker

def trainListener(problems, speaker, hiddenNodes=[24]):
  """Speaker is in functional form. Returns ANN"""
  dataset = listenerTrainingExamplesFromInstances(problems, speaker)
  (listener, trainer) = trainAgent(dataset, 300, hiddenNodes)
  return listener



#### convience functions ####

def featuresToString(features):
  """Pretty print features.
    hat, glasses, moustache: ^*_ |^ _| 
    everyone gets 3 spaces, one character per space
    Desired output: 
  """
  face0 = features[0:3]
  face1 = features[3:6]
  face2 = features[6:9]
  return "%s %s %s" % (faceToString(face0), faceToString(face1), faceToString(face2))

def faceToString(face):
  """ face is a 3 length np array """
  c1=c2=c3 = ' '
  if face[0]:
    c1 = '^'  
  if face[1]:
    c2 = '*'
  if face[2]:
    c3 = '_'
  return "<%s%s%s>" % (c1, c2, c3)

def utteranceToString(utterance):
  """utterance is a 3 length 1-hot array"""
  if utterance[0]:
    return '^'
  elif utterance[1]:
    return '*'
  elif utterance[2]:
    return '_'
  else:
    return 'ERROR'  

def utteranceIdToString(idx):
  if idx == 0:
    return '^'
  elif idx == 1:
    return '*'
  elif idx == 2:
    return '_'
  else:
    return 'ERROR'


def trainAllAgents(dataPath, outPath, hiddenNodes, nIterations):
  trnProblems = loadFacesInstances(dataPath)
  listeners = []
  speakers = []
  l1Listener = trainListener(trnProblems, literalSpeaker)
  l1Speaker = trainSpeaker(trnProblems, literalListener)
  listeners.append(l1Listener)
  speakers.append(l1Speaker)
  print "[trainAllAgents] Output %s hidden: %s iter: %d" % (outPath, hiddenNodes, nIterations)
  for i in range(1, nIterations):
    print "[trainAllAgents] Training iteration", i
    speaker = trainSpeaker(trnProblems, listenerANNToFunction(listeners[-1]), hiddenNodes)
    speakers.append(speaker)
    listener = trainListener(trnProblems, speakerANNToFunction(speakers[-1]), hiddenNodes)
    listeners.append(listener)
  output = (listeners, speakers)
  f = open(outPath, 'w')
  pickle.dump(output, f)
  f.close()

  

def loadAllAgents(path):
  """Loads all agents from a pickle file. Order is:
     (l1Listener, l1Speaker, l2Listener, l2Speaker, ....)
  """
  f = open(path)
  output = pickle.load(f)
  f.close()
  return output

  
def syntheticHiddenPlot():
  """ Evaluate a variety of hidden layer agents"""
  matplotlib.rcParams.update({'font.size' : 20})
  lw = 3
  plt.hold(True)
  levelInstances = [loadFacesInstances('../../data/facesInstances-%d.csv' % level) for level in [0,1,2]]
  sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
  nModels = 10 # numbered 1 to 10
  agents = [] # will be an array of arrays, one entry per hidden node, one entry per training iteration
  # load the agents
  for size in sizes:
    sizeAgents = []
    for agentNum in range(1,nModels + 1):
      (listeners, speakers) = loadAllAgents('../../data/cogsci/agents-%d-%d.pickle' % (size, agentNum))
      sizeAgents.append(listeners)
    agents.append(sizeAgents)
  # loop over levels, then over model sizes, then over agents..
  for (instances, lineColor) in zip(levelInstances, colors): # for each level
    dataset = goldListenerTrainingExamplesFromInstances(instances)
    hiddenLayerAccuracies = [] # average accuracy for each hidden layer
    hiddenLayerScores = []
    yerrs = []
    for (allListeners, size) in zip(agents, sizes): # for each # of hidden layers
      sizeAccuracies = [] # accuracies for each independent trial for this # of hidden nodes and this level of problem. will be averaged.
      sizeScores = []
      for listeners in allListeners:
        lastListener = listeners[3] 
        (correct, activations, scores) = evalListenerOnClassificationDataset(lastListener, dataset)
        sizeAccuracies.append(float(correct) / len(scores))
        sizeScores.append(scores)
      averageAccuracy = np.array(sizeAccuracies).mean()
      hiddenLayerAccuracies.append(averageAccuracy)
      hiddenLayerScores.append(sizeScores)
      interval = boot.ci(np.array(sizeScores), np.average) 
      err = (interval[1] - interval[0])/2.0
      yerrs.append(err)
    plt.errorbar(sizes, hiddenLayerAccuracies, yerr=yerrs, linewidth=lw, color=lineColor)
  plt.title('ANN Accuracy by Size of Hidden Layer')
  plt.axis([0, sizes[-1], 0, 1])
  plt.xlabel('Number of Hidden Nodes')
  plt.ylabel('Listener Accuracy')
  legendTitles = ['Level 0', 'Level 1', 'Level 2']
  plt.legend(legendTitles, loc='lower right')
  plt.savefig('hiddenSynthetic.pdf', format='pdf')
  plt.show()

def scalesHiddenPlot(name='scales'):
  matplotlib.rcParams.update({'font.size' : 20})
  lw = 3
  plt.hold(True)
  if name == 'scalesPlus':
    experimentName = 'Complex'
    nLevels = 3
    leveledFcData = turk.readScalesProblems('../../data/scale_plus_6stimuli_3levels_no_fam_24_january_SCAL.csv', name)
  elif name == 'scales':
    experimentName = 'Simple'
    nLevels = 2
    leveledFcData = turk.readScalesProblems('../../data/scales_6stimuli_3levels_no_fam_25_january_OSCA.csv', name)
  else:
    print '[forcedChoiceExperiments] Unknown experiment name: ', name
  sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
  nModels = 10 # numbered 1 to 10
  agents = [] # will be an array of arrays, one entry per hidden node, one entry per training iteration
  # load the agents
  for size in sizes:
    sizeAgents = []
    for agentNum in range(1,nModels + 1):
      (listeners, speakers) = loadAllAgents('../../data/cogsci/agents-%d-%d.pickle' % (size, agentNum))
      sizeAgents.append(listeners)
    agents.append(sizeAgents)
  for (levelProblems, lineColor) in zip(leveledFcData, colors):
    dataset = forcedChoiceProblemsToDataset(levelProblems)
    hiddenLayerAccuracies = []    
    hiddenLayerScores = []
    yerrs = []
    for (allListeners, size) in zip(agents, sizes): # for each # of hidden layers
      sizeAccuracies = [] # accuracies for each independent trial for this # of hidden nodes and this level of problem. will be averaged.
      sizeScores = []
      for listeners in allListeners:
        lastListener = listeners[3] 
        (correct, activations, scores) = evalListenerOnClassificationDataset(lastListener, dataset)
        sizeAccuracies.append(float(correct) / len(scores))
        sizeScores.append(scores)
      averageAccuracy = np.array(sizeAccuracies).mean()
      hiddenLayerAccuracies.append(averageAccuracy)
      hiddenLayerScores.append(sizeScores)
      interval = boot.ci(np.array(sizeScores), np.average) 
      err = (interval[1] - interval[0])/2.0
      yerrs.append(err)
    plt.errorbar(sizes, hiddenLayerAccuracies, yerr=yerrs, linewidth=lw, color=lineColor)
  plt.axis([0, sizes[-1], 0, 1])
  plt.title('ANN Accuracy on the %s Condition' % experimentName)
  plt.xlabel('Number of Hidden Nodes')
  plt.ylabel('Average Accuracy')
  plt.legend(['Level %d' % i for i in range(nLevels)], loc='lower right')
  plt.savefig('hidden%s.pdf' % name, format='pdf')
  plt.show()


def ibrSyntheticPlot():
  matplotlib.rcParams.update({'font.size' : 20})
  lw = 3
  plt.hold(True)
  for level, lineColor in zip([0,1,2], colors):
    # load data
    f = open('../../data/Rsynthetic-%d.txt' % level)
    accuracies = [] # array of arrays
    for line in f:
      probs = [float(x) for x in str.split(line)]
      accuracies.append(probs) 
    # now plot them. well, need to average them.
    sums = np.zeros(len(accuracies[0]))
    for depth in range(len(accuracies[0])):
      for result in range(len(accuracies)):
        sums[depth] = sums[depth] + accuracies[result][depth]
    averages = [x / len(accuracies) for x in sums] 
    plt.plot(range(0, len(averages)), averages, color=lineColor, linewidth=lw)
  plt.title('IBR Accuracy on Synthetic Data')
  plt.ylabel('Listener Accuracy')
  plt.xlabel('Depth of IBR Recursion')
  plt.axis([0, 20, 0, 1])
  plt.legend(['Level %d' % i for i in [0,1,2]], loc='lower right')
  plt.savefig('IBRsynthetic.pdf', format='pdf')
  plt.show()

def syntheticPlot(allAccuracies, allScores, outFile, title, errorBars=False, overall=False):
  """Generate figure of accuracy accross listeners and datasets.
     accuracy is an array of np arrays. each np array has the accuracy for a given level of each model, where the 0th is the literal one.
     If overall == True, use the last entry labeled as Overall. Otherwise, label the scores by their level name.
  """
  matplotlib.rcParams.update({'font.size' : 20})
  lw = 3
  plt.hold(True)
  for levelAccuracies, levelScores, lineColor in zip(allAccuracies, allScores, colors):
    if errorBars:
      yerrs = []
      for scores in levelScores: # one per each level
        if np.array(scores).all():
          yerrs.append(0)
        else:
          interval = boot.ci(np.array(scores), np.average) 
          err = (interval[1] - interval[0])/2.0
          yerrs.append(err)
      plt.errorbar(range(len(levelAccuracies)), levelAccuracies, yerr=yerrs, linewidth=lw, color=lineColor)
    else:
      plt.plot(levelAccuracies, linewidth=lw, marker='o', color=lineColor)
  nListeners = len(allAccuracies[0]) # number of models
  nLevels = len(allAccuracies) # types of problems
  plt.axis([0, nListeners - 1, 0, 1])
  plt.ylabel('Listener Accuracy')
  plt.xlabel('Training Iterations')
  if overall:
    legendTitles = ['Level %d' % level for level in range(nLevels - 1)]
    legendTitles.append('Overall')
  else:
    legendTitles = ['Level %d' % level for level in range(nLevels)]
    
  plt.legend(legendTitles, loc='lower right')
  plt.title(title)
  plt.savefig(outFile, format='pdf')
  plt.show()

def forcedChoicePlot(listenerAccuracies, listenerScores, mturkAccuracies, mturkScores, outFile, title, errorBars=False):
  """listenerAccuracies is an array of accuracy arrays, one per problem level.
     mturkAccuracies is a 1-d array of mturk accuracies on each problem level. 
  """
  matplotlib.rcParams.update({'font.size' : 20})
  lw = 4
  plt.hold(True)
  nListeners = len(listenerAccuracies)
  nIterations = len(listenerAccuracies[0]) - 1
  plt.axis([0, nIterations, 0, 1])
  plt.ylabel('Listener Accuracy')
  plt.xlabel('Training Iterations')
  for levelAccuracies, levelScores, lineColor in zip(listenerAccuracies, listenerScores, colors):
    if errorBars: 
      yerrs = []
      for scores in levelScores: 
        if np.array(scores).all():
          yerrs.append(0)
        else:
          interval = boot.ci(np.array(scores), np.average)
          err = (interval[1] - interval[0]) / 2.0
          yerrs.append(err)
      plt.errorbar(range(len(levelAccuracies)), levelAccuracies, yerr=yerrs, linewidth=lw, color=lineColor)
      print lineColor
      print levelAccuracies
    else:
      plt.plot(levelAccuracies, linewidth=lw, marker='o', color=lineColor) 
  listenerTitles = ['Level %d' % level for level in range(nListeners)]
  plt.legend(listenerTitles, loc='lower right')
  plt.title(title)
  plt.savefig(outFile, format='pdf')
  plt.show()


def syntheticMultiExperiments(agentPrefix, nAgents, overall=False, errorBars=False):
  """ agentPrefix: something like agents-20-
      nAgents: usually 10, then forms agents-20-1.pickle, agents-20-2.pickle, etc.
  """
  agents = [] # list of list of listeners, where agents[i] is each independent trained list of agents, so for instance agents[0][0] would be the 1st agent trained on 1 iteration.
  for i in range(1, nAgents + 1): # ugh, numbered then agents-20-1 and so on.
    agentPath = '%s%d.pickle' % (agentPrefix, i)
    (listeners, speakers) = loadAllAgents(agentPath)
    agents.append(listeners)
  if overall:
    levelInstances = [loadFacesInstances('../../data/facesInstances-%d.csv' % level) for level in [0,1,2,3]]
  else:
    levelInstances = [loadFacesInstances('../../data/facesInstances-%d.csv' % level) for level in [0,1,2]]
  allAccuracies = [] # will end up with one per level
  allActivations = []
  allScores = []
  for level, instances in enumerate(levelInstances):
    dataset = goldListenerTrainingExamplesFromInstances(instances)
    nProblems = len(dataset)
    (literalCorrect, literalTargets, literalScores) = evalLiteralListenerOnClassificationDataset(literalListener, dataset)
    levelAccuracies = [float(literalCorrect) / nProblems]
    levelActivations = [literalTargets]
    levelScores = [literalScores]
    nIterations = len(agents[0]) # number of training iterations
    for trainingIdx in range(nIterations):
      iterationScores = []
      iterationActivations = []
      for agentIdx in range(nAgents):
        listener = agents[agentIdx][trainingIdx] 
        (correct, activations, scores) = evalListenerOnClassificationDataset(listener, dataset)
        iterationScores.append(scores) # perhaps need to do a flatmap here?
        iterationActivations.append(activations)
      flattenedScores = [score for agentScores in iterationScores for score in agentScores]
      levelScores.append(flattenedScores)
      iterationAccuracy = np.array(flattenedScores).mean()
      levelAccuracies.append(iterationAccuracy)
      levelActivations.append(np.array([activation for agentActivations in iterationActivations for activation in agentActivations]))
    allAccuracies.append(np.array(levelAccuracies))
    allActivations.append(levelActivations)
    allScores.append(np.array(levelScores))
  syntheticPlot(allAccuracies, allScores, 'synthetic.pdf', 'ANN Accuracy on Synthetic Data', errorBars=errorBars, overall=overall)

def batchSyntheticExperiments(agentPrefix, nAgents, overall=False, errorBars=True):
  for i in range(1, nAgents + 1): # ugh, numbered then agents-20-1 and so on.
    agentPath = '%s%d.pickle' % (agentPrefix, i)
    (listeners, speakers) = loadAllAgents(agentPath)
    if overall:
      levelInstances = [loadFacesInstances('../../data/facesInstances-%d.csv' % level) for level in [0,1,2,3]]
    else:
      levelInstances = [loadFacesInstances('../../data/facesInstances-%d.csv' % level) for level in [0,1,2]]
    allAccuracies = []
    allActivations = []
    allScores = []
    for level, instances in enumerate(levelInstances):
      dataset = goldListenerTrainingExamplesFromInstances(instances)
      nProblems = len(dataset)
      (literalCorrect, literalTargets, literalScores) = evalLiteralListenerOnClassificationDataset(literalListener, dataset)
      levelAccuracies = [float(literalCorrect) / nProblems]
      levelActivations = [literalTargets]
      levelScores = [literalScores]
      for listener in listeners:
        (correct, activations, scores) = evalListenerOnClassificationDataset(listener, dataset)
        levelAccuracies.append(float(correct) / nProblems)
        levelActivations.append(activations)
        levelScores.append(scores)
      allAccuracies.append(np.array(levelAccuracies))
      allActivations.append(np.array(levelActivations))
      allScores.append(np.array(levelScores))
    syntheticPlot(allAccuracies, allScores, '%s%d-synthetic.pdf' % (agentPrefix,i), 'Model Performance on Synthetic Data', overall=overall, errorBars=errorBars)
    

def syntheticExperiments(agentPath, overall=False, errorBars=False):
  """Evaluate listener models on all of the "synthetic" data.
     Want numbers for each level of problem as well. Report both raw # correct and also accuracy %s
     If overall == true, load an extra level, which is the combined results.
  """
  (listeners, speakers) = loadAllAgents(agentPath)
  if overall:
    levelInstances = [loadFacesInstances('../../data/facesInstances-%d.csv' % level) for level in [0,1,2,3]]
  else:
    levelInstances = [loadFacesInstances('../../data/facesInstances-%d.csv' % level) for level in [0,1,2]]
  allAccuracies = []
  allActivations = []
  allScores = []
  for level, instances in enumerate(levelInstances):
    dataset = goldListenerTrainingExamplesFromInstances(instances)
    nProblems = len(dataset)
    (literalCorrect, literalTargets, literalScores) = evalLiteralListenerOnClassificationDataset(literalListener, dataset)
    levelAccuracies = [float(literalCorrect) / nProblems]
    levelActivations = [literalTargets]
    levelScores = [literalScores]
    for listener in listeners:
      (correct, activations, scores) = evalListenerOnClassificationDataset(listener, dataset)
      levelAccuracies.append(float(correct) / nProblems)
      levelActivations.append(activations)
      levelScores.append(scores)
    allAccuracies.append(np.array(levelAccuracies))
    allActivations.append(np.array(levelActivations))
    allScores.append(np.array(levelScores))
  syntheticPlot(allAccuracies, allScores, 'synthetic.pdf', 'Model Performance on Synthetic Data', overall=overall, errorBars=errorBars)

def forcedChoiceMultiExperiments(agentPrefix, nAgents, plotPath, name='scales', errorBars=False):
  if name == 'scalesPlus':
    condition = 'Complex'
    leveledFcData = turk.readScalesProblems('../../data/scale_plus_6stimuli_3levels_no_fam_24_january_SCAL.csv', name)
  elif name == 'scales':
    condition = 'Simple'
    leveledFcData = turk.readScalesProblems('../../data/scales_6stimuli_3levels_no_fam_25_january_OSCA.csv', name)
  else:
    print '[forcedChoiceExperiments] Unknown experiment name: ', name
  agents = [] # list of list of listeners, where agents[i] is each independent trained list of agents, so for instance agents[0][0] would be the 1st agent trained on 1 iteration.
  for i in range(1, nAgents + 1): # ugh, numbered then agents-20-1 and so on.
    agentPath = '%s%d.pickle' % (agentPrefix, i)
    (listeners, speakers) = loadAllAgents(agentPath)
    agents.append(listeners)
  allAccuracies = []
  allActivations = []
  allScores = []
  for level, fcData in enumerate(leveledFcData):
    dataset = forcedChoiceProblemsToDataset(fcData)
    (literalCorrect, literalTargets, literalScores) = evalLiteralListenerOnClassificationDataset(literalListener, dataset)
    levelAccuracies = [float(literalCorrect) / len(dataset)]
    levelActivations = [literalTargets]
    levelScores = [literalScores]
    nIterations = len(agents[0])
    for trainingIdx in range(nIterations):
      iterationScores = []
      iterationActivations = []
      for agentIdx in range(nAgents):
        listener = agents[agentIdx][trainingIdx]
        (correct, activations, scores) = evalListenerOnClassificationDataset(listener, dataset)
        iterationScores.append(scores)
        iterationActivations.append(activations)
      flattenedScores = [scores for agentScores in iterationScores for scores in agentScores] 
      levelScores.append(flattenedScores)
      levelAccuracies.append(np.array(flattenedScores).mean())
      levelActivations.append(np.array([activation for agentActivations in iterationActivations for activation in agentActivations]))
    allAccuracies.append(np.array(levelAccuracies))
    allActivations.append(levelActivations)
    allScores.append(np.array(levelScores))
  forcedChoicePlot(allAccuracies, allScores, [], [], plotPath, 'ANN Accuracy on the %s Condition' % condition, errorBars=errorBars)

def RPlot(fName, outName):
  """Create the plots for IBR performance on the scales and scales+ problems. first, read the files."""
  f = open(fName)
  allScores = []
  for line in f:
    levelScores = [float(entry) for entry in str.split(line)]
    allScores.append(levelScores) 
  f.close()
  matplotlib.rcParams.update({'font.size' : 20})
  lw = 4
  plt.hold(True)
  print allScores
  for levelScores, lineColor in zip(allScores, colors):
    plt.plot(range(len(levelScores)), levelScores, linewidth=lw, color=lineColor)
    #lw = 3
  plt.legend(['Level %d' % i for i in range(len(allScores))], loc='lower right')
  if(len(allScores) == 2):
    plt.title('IBR Accuracy on Simple Condition')
  else:
    plt.title('IBR Accuracy on Complex Condition')
  plt.xlabel('Depth of IBR Recursion')
  plt.ylabel('Listener Accuracy')
  plt.axis([0, 20, 0, 1])
  plt.savefig(outName, format='pdf')
  plt.show()

def forcedChoiceExperiments(agentPath, plotPath, name='scales', errorBars=False):
  """ name can be 'scales' or 'scalesPlus' """
  if name == 'scalesPlus':
    leveledFcData = turk.readScalesProblems('../../data/scale_plus_6stimuli_3levels_no_fam_24_january_SCAL.csv', name)
  elif name == 'scales':
    leveledFcData = turk.readScalesProblems('../../data/scales_6stimuli_3levels_no_fam_25_january_OSCA.csv', name)
  else:
    print '[forcedChoiceExperiments] Unknown experiment name: ', name
  (listeners, speakers) = loadAllAgents(agentPath)
  listenerAccuracies = []
  listenerActivations = []
  listenerScores = []
  mturkAccuracies = []
  mturkScores = [] 
  for level, fcData in enumerate(leveledFcData):
    dataset = forcedChoiceProblemsToDataset(fcData)
    nProblems = len(dataset)
    levelAccuracies = []
    levelActivations = []
    levelScores = []

    (mturkAccuracy, turkScores) = evalTurkForcedChoices(fcData)
    turkChoices = [problem[3] for problem in fcData]
    mturkAccuracies.append(mturkAccuracy)
    mturkScores.append(turkScores)

    (literalCorrect, literalTargets, literalScores) = evalLiteralListenerOnClassificationDataset(literalListener, dataset)
    levelAccuracies.append(float(literalCorrect) / nProblems)
    levelActivations.append(literalTargets)
    levelScores.append(literalScores)
    for listener in listeners:
      (correct, activations, scores) = evalListenerOnClassificationDataset(listener, dataset)
      levelAccuracies.append(float(correct) / nProblems)
      levelActivations.append(activations)
      levelScores.append(scores)
    listenerAccuracies.append(np.array(levelAccuracies))
    listenerActivations.append(np.array(levelActivations))
    listenerScores.append(np.array(levelScores))
  forcedChoicePlot(listenerAccuracies, listenerScores, mturkAccuracies, mturkScores, plotPath, 'Model Performance on Forced Choice Data', errorBars=errorBars)

def evalListenerOnClassificationDataset(listener, fcProblems):
  """Evaluate a listener agent on the forced choice dataset.
     Here fcProblems is a ClassificationDataset, where problem[0] is concatenated features, utterance, 
     and problem[1] is target.
     returns number correct, raw model activations, and also an array of if each problem is correct (1/0)
  """
  nCorrect = 0
  allActivations = []
  correct = []
  for problem in fcProblems:
    activations = listener.activate(problem[0])
    allActivations.append(activations)
    activationMax = activations.argmax()
    target = one_to_n(activationMax, 3)
    if (target == problem[1]).all():
      nCorrect += 1
      correct.append(1)
    else:
      correct.append(0) 
  return (nCorrect, allActivations, correct)

def evalSpeakerOnClassificationDataset(speaker, problems):
  """Evaluate a listener agent on the forced choice dataset.
     Here fcProblems is a ClassificationDataset, where problem[0] is concatenated features, utterance, 
     and problem[1] is target.
  """
  nCorrect = 0
  allActivations = []
  for features, goldUtterance in problems:
    activations = speaker.activate(features)
    allActivations.append(activations)
    activationMax = activations.argmax()
    utterance = one_to_n(activationMax, 3)
    if (utterance == goldUtterance).all():
      nCorrect +=1
  return nCorrect, allActivations 

def evalLiteralListenerOnClassificationDataset(listener, fcProblems):
  """ fcProblems is a ClassificationDataset """
  nCorrect = 0
  allTargets = []
  scores = []
  for problem in fcProblems:
    allFeatures = problem[0]
    features = allFeatures[:9] 
    utterance = allFeatures[9:]
    predictedTarget = listener(features, utterance)
    allTargets.append(predictedTarget)
    if (predictedTarget == problem[1]).all(): 
      nCorrect += 1
      scores.append(1)
    else:
      scores.append(0)
  return nCorrect, allTargets, scores

def evalTurkForcedChoices(fcData):
  """ Evaluate the turkers on the forced choice data. Returns accuracy. """
  nCorrect = 0
  scores = []
  for problem in fcData:
    if (problem[3] == problem[2]).all():
      nCorrect += 1
      scores.append(1)
    else:
      scores.append(0)
  accuracy = float(nCorrect) / len(fcData)
  return accuracy, scores

def forcedChoiceProblemsToDataset(fcProblems):
  dataset = ClassificationDataSet(numFeatures, 3, nb_classes=3, class_labels=['face%d' % i for i in range(3)])
  for problem in fcProblems:
    dataset.addSample(np.concatenate([problem[0], problem[1]]), problem[2])
  return dataset

def loadFacesInstances(fname):
  """Loads specific problem instances in the form of (problem, utterance, target)"""
  f = open(fname)
  lines = [x for x in csv.reader(f)]
  f.close()
  instances = []
  for line in lines:
    floats = [float(item) for item in line]
    problem = np.array(floats[:9])
    utterance = np.array(floats[9:12])
    target = np.array(floats[12:])
    instances.append((problem, utterance, target))
  return instances 

def usage():
  print "./discrim_ibr.py train <outFile> <iterations> <hidden nodes+>"
  print "./discrim_ibr.py eval <agentFile>"
  print "./discrim_ibr.py ibr" #produces ibr plot
  print "./discrim_ibr.py hidden" #produces hidden layer plot
  print "./discrim_ibr.py multieval <agentPrefix> <nAgents> " # ex ./discrim_ibr.py multieval data/cogsci/agents-20- 10
  
if __name__ == '__main__':
  if len(sys.argv) > 2 and sys.argv[1] == 'train' and len(sys.argv) >= 5:
    outPath = sys.argv[2]
    nIterations = int(sys.argv[3])
    hiddenNodes = [int(n) for n in sys.argv[4:]]
    trainAllAgents('../../data/facesInstances.csv', outPath, hiddenNodes, nIterations)
  elif len(sys.argv) > 2 and sys.argv[1] == 'eval' and len(sys.argv) == 3:
    syntheticExperiments(sys.argv[2], overall=False, errorBars=True)
    forcedChoiceExperiments(sys.argv[2], 'scales.pdf', name='scales', errorBars=False)
    forcedChoiceExperiments(sys.argv[2], 'scalesPlus.pdf', name='scalesPlus', errorBars=False)
  elif len(sys.argv) > 3 and sys.argv[1] == 'multieval': 
    syntheticMultiExperiments(sys.argv[2], int(sys.argv[3]), overall=False, errorBars=True)
    forcedChoiceMultiExperiments(sys.argv[2], int(sys.argv[3]), 'scales.pdf', name='scales', errorBars=True)
    forcedChoiceMultiExperiments(sys.argv[2], int(sys.argv[3]), 'scalesPlus.pdf', name='scalesPlus', errorBars=True)
  elif len(sys.argv) == 2 and sys.argv[1] == 'ibr':
    ibrSyntheticPlot()
    RPlot('../../data/Rscales.txt', 'IBRscales.pdf')
    RPlot('../../data/RscalesPlus.txt', 'IBRscalesPlus.pdf')
  elif len(sys.argv) == 2 and sys.argv[1] == 'hidden':
    syntheticHiddenPlot()
    scalesHiddenPlot('scalesPlus')
    scalesHiddenPlot('scales')
  elif len(sys.argv) > 3 and sys.argv[1] == 'batch':
    # make a bunch of plots at once
    batchSyntheticExperiments(sys.argv[2], int(sys.argv[3]))
  else:
    print "Unknown invocation:" , ' '.join(sys.argv)
    usage()
