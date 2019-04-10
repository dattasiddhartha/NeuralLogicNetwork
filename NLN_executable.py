# initialization
import pandas as pd

data = pd.read_csv("initialdata.csv")

encodedData = pd.DataFrame()
for col in data.columns:
    df = pd.get_dummies(data[col]) # one hot encoding
    for newcol in df.columns:
        name = str(col)+"_"+str(newcol)
        encodedData[name] = df[newcol]
        
# single output column
encodedDataComplete = encodedData.drop("Default_N", axis=1)

xCol = encodedDataComplete.columns[0:-1]
yCol = encodedDataComplete.columns[-1]
outputData = encodedDataComplete[yCol]
encodedData = encodedDataComplete[xCol]

# logic gate learning

import numpy
import scipy.special
import glob
import scipy.misc
class neuralNetwork:
    def __init__(self, inputNodes, hiddenOneNodes, hiddenTwoNodes, hiddenThreeNodes, finalNodes, alpha):
        self.inputNodes = inputNodes
        self.hiddenOneNodes = hiddenOneNodes
        self.hiddenTwoNodes = hiddenTwoNodes
        self.hiddenThreeNodes = hiddenThreeNodes
        self.finalNodes = finalNodes
        self.alpha = alpha
        self.weightsInputHidden = numpy.random.normal(0.0, pow(self.hiddenOneNodes, -0.5),(self.hiddenOneNodes,self.inputNodes))
        self.weightsHiddenOneHiddenTwo = numpy.random.normal(0.0, pow(self.hiddenTwoNodes,-0.5),(self.hiddenTwoNodes,self.hiddenOneNodes))
        self.weightsHiddenTwoHiddenThree = numpy.random.normal(0.0, pow(self.hiddenThreeNodes,-0.5),(self.hiddenThreeNodes,self.hiddenTwoNodes))
        self.weightsHiddenOutput = numpy.random.normal(0.0, pow(self.hiddenOneNodes,-0.5),(self.finalNodes, self.hiddenThreeNodes))
        pass
    def train(self, inputs, target):
        inputs = numpy.array(inputs, ndmin=2).T
        target = numpy.array(target, ndmin=2).T
        hiddenInput = numpy.dot(self.weightsInputHidden,inputs)
        hiddenOneOutput = self.sigmoid(hiddenInput)
        hiddenTwoInput = numpy.dot(self.weightsHiddenOneHiddenTwo,hiddenOneOutput)
        hiddenTwoOutput = self.sigmoid(hiddenTwoInput)
        hiddenThreeInput = numpy.dot(self.weightsHiddenTwoHiddenThree,hiddenTwoOutput)
        hiddenThreeOutput = self.sigmoid(hiddenThreeInput)
        finalInput = numpy.dot(self.weightsHiddenOutput,hiddenThreeOutput)
        finalOutput = self.sigmoid(finalInput)
        outputError = target - finalOutput
        hiddenOutputError = numpy.dot(self.weightsHiddenOutput.T, outputError)
        hiddenThreeHiddenTwoError = numpy.dot(self.weightsHiddenTwoHiddenThree.T, hiddenOutputError)
        hiddenTwoHiddenOneError = numpy.dot(self.weightsHiddenOneHiddenTwo.T, hiddenThreeHiddenTwoError)
        hiddenInputError = numpy.dot(self.weightsInputHidden.T, hiddenTwoHiddenOneError)
        self.weightsHiddenOutput += self.alpha * numpy.dot((outputError * finalOutput * (1.0 - finalOutput)),numpy.transpose(hiddenThreeOutput))
        self.weightsHiddenTwoHiddenThree += self.alpha * numpy.dot((hiddenOutputError * hiddenThreeOutput * (1.0 - hiddenThreeOutput)),numpy.transpose(hiddenTwoOutput))
        self.weightsHiddenOneHiddenTwo += self.alpha * numpy.dot((hiddenThreeHiddenTwoError * hiddenTwoOutput * (1.0 - hiddenTwoOutput)),numpy.transpose(hiddenOneOutput))
        self.weightsInputHidden += self.alpha * numpy.dot((hiddenTwoHiddenOneError * hiddenOneOutput * (1.0 - hiddenOneOutput)),numpy.transpose(inputs))        
        pass
    def query(self, inputs):
        inputs = numpy.array(inputs, ndmin=2).T
        hiddenInput = numpy.dot(self.weightsInputHidden,inputs)
        hiddenOneOutput = self.sigmoid(hiddenInput)
        hiddenTwoInput = numpy.dot(self.weightsHiddenOneHiddenTwo,hiddenOneOutput)
        hiddenTwoOutput = self.sigmoid(hiddenTwoInput)
        hiddenThreeInput = numpy.dot(self.weightsHiddenTwoHiddenThree,hiddenTwoOutput)
        hiddenThreeOutput = self.sigmoid(hiddenThreeInput)
        finalInput = numpy.dot(self.weightsHiddenOutput,hiddenThreeOutput)
        finalOutput = self.sigmoid(finalInput)
        return finalOutput
        pass
    def sigmoid(self, x):
        return scipy.special.expit(x)
        pass
        
#AND
nnAND = neuralNetwork(2,12,36,12,1,0.1)
print('Before training')
print(nnAND.query([0,0]))
print(nnAND.query([0,1]))
print(nnAND.query([1,0]))
print(nnAND.query([1,1]))
print("Training...")
for i in range(0, 10000):
    nnAND.train([0,0],[0])
    nnAND.train([0,1],[0])
    nnAND.train([1,0],[0])
    nnAND.train([1,1],[1])
print("Done")
print(nnAND.query([0,0]))
print(nnAND.query([0,1]))
print(nnAND.query([1,0]))
print(nnAND.query([1,1]))

#OR
nnOR = neuralNetwork(2,12,36,12,1,0.1)
print('Before training')
print(nnOR.query([0,0]))
print(nnOR.query([0,1]))
print(nnOR.query([1,0]))
print(nnOR.query([1,1]))
print("Training...")
for i in range(0, 10000):
    nnOR.train([0,0],[0])
    nnOR.train([0,1],[1])
    nnOR.train([1,0],[1])
    nnOR.train([1,1],[1])
print("Done")
print(nnOR.query([0,0]))
print(nnOR.query([0,1]))
print(nnOR.query([1,0]))
print(nnOR.query([1,1]))

#NOT
nnNOT = neuralNetwork(1,12,36,12,1,0.1)
print('Before training')
print(nnNOT.query([0]))
print(nnNOT.query([1]))
print("Training...")
for i in range(0, 10000):
    nnNOT.train([0],[1])
    nnNOT.train([1],[0])
print("Done")
print(nnNOT.query([0]))
print(nnNOT.query([1]))

#NAND
nnNAND = neuralNetwork(2,12,36,12,1,0.1)
print('Before training')
print(nnNAND.query([0,0]))
print(nnNAND.query([0,1]))
print(nnNAND.query([1,0]))
print(nnNAND.query([1,1]))
print("Training...")
for i in range(0, 10000):
    nnNAND.train([0,0],[1])
    nnNAND.train([0,1],[1])
    nnNAND.train([1,0],[1])
    nnNAND.train([1,1],[0])
print("Done")
print(nnNAND.query([0,0]))
print(nnNAND.query([0,1]))
print(nnNAND.query([1,0]))
print(nnNAND.query([1,1]))

#XOR
nnXOR = neuralNetwork(2,12,36,12,1,0.1)
print('Before training')
print(nnXOR.query([0,0]))
print(nnXOR.query([0,1]))
print(nnXOR.query([1,0]))
print(nnXOR.query([1,1]))
print("Training...")
for i in range(0, 10000):
    nnXOR.train([0,0],[0])
    nnXOR.train([0,1],[1])
    nnXOR.train([1,0],[1])
    nnXOR.train([1,1],[0])
print("Done")
print(nnXOR.query([0,0]))
print(nnXOR.query([0,1]))
print(nnXOR.query([1,0]))
print(nnXOR.query([1,1]))

# NLN Training on use case

from random import randint
import itertools
from tqdm import tnrange, tqdm_notebook, tqdm
from time import sleep
# for index, row in encodedData.iterrows():
# for i in tnrange(3, desc='1st loop'):
#     for j in tqdm_notebook(range(100), desc='2nd loop'):
#         sleep(0.01)

logicName = ['AND', 'OR', 'NAND', 'XOR']
logicNN = [nnAND, nnOR, nnNAND, nnXOR] # nnNOT does not work at the moment

logicHistory = []
varHistory = []
logicnameHistory = []

varLogs = []
logicLogs = []
logicnameLogs = []
accuracyLogs = []

varName = list(encodedData.columns)
placeholderList = []

# select variables
n = 10000
for i in tnrange(n): # n iterations

    for var in varName:

        placeholderList = [item for item in varName if item!=var]

        varHistory.append(var)

        for var2 in placeholderList:
            placeholderList = [item for item in placeholderList if item!=var2]

            # variables selected > select logic gate
            randomIndex = randint(0, len(logicName)-1)
            gate = logicNN[randomIndex]

            input1 = encodedData[var][0]
            input2 = encodedData[var2][0]

            output = gate.query([input1, input2])

            logicHistory.append(gate); logicnameHistory.append(logicName[randomIndex])
            varHistory.append(var2)
           

            # use output with next var in placeholderlist
            for varZ in placeholderList:
                placeholderList = [item for item in placeholderList if item!=varZ]
                randomIndex = randint(0, len(logicName)-1)
                gate = logicNN[randomIndex]
                inputZ = encodedData[varZ][0]

                output = gate.query([output[0][0], inputZ])

                logicHistory.append(gate); logicnameHistory.append(logicName[randomIndex])
                varHistory.append(varZ)

                # check output
                if varZ == list(encodedData.columns)[-1]:

                    if round(output[0][0]) == outputData[0]: # output match -- now check how many rows of data does the logic model fit

                        # execute for all rows, generate accuracy score

                        for index, row in encodedDataComplete.iterrows():
                            correctCount = 0
                            intCounter = 0
                            v1 = row[varHistory[intCounter]]
                            while intCounter < len(varHistory)-1:

                                lg = logicHistory[intCounter]
                                intCounter +=1
                                v2 = row[varHistory[intCounter]]
    #                             print(lg)
                                otpt = lg.query([v1, v2])[0][0]
                                v1 = otpt

                            if round(otpt) == row[list(encodedDataComplete.columns)[-1]]:
                                correctCount+=1

                        # store accuracy score, store history logs
                        accuracy = correctCount / float(len(encodedDataComplete))
                        accuracyLogs.append(accuracy)
                        varLogs.append(varHistory)
                        logicLogs.append(logicHistory)
                        logicnameLogs.append(logicnameHistory)

        logicHistory = []
        logicnameHistory = []
        varHistory = []

    print("Iteration " + str(i+1))
            

logsDF = pd.DataFrame()
logsDF['varStructure'] = varLogs
logsDF['logicStructure'] = logicnameLogs
logsDF['logicMachine'] = logicLogs
logsDF['accuracy'] = accuracyLogs
logsDF = logsDF.sort_values(by=['accuracy'], ascending=False)

logsDF.to_csv("logs_NLN.csv")