# Adolfo Alvarez Jr

import random
from annMath import ANNMath

"""
This class defines a simple artificial neural network (ANN) structure with an input layer, a hidden layer,
and an output layer.
"""

class SimpleANN:

    # This method creates the ANN structure and initializes random weights and biases
    def __init__(self, inputNodes: int = 2, hiddenNodes: int = 5, outputNodes: int = 3, learningRate: float = 0.1):

        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNodes = outputNodes
        self.learningRate = learningRate

        self.weightsInputHidden = []
        self.weightsHiddenOutput = []
        self.biasHidden = []
        self.biasOutput = []

        self.initializeWeights()
        self.initializeBiases()

    # This method creates random weights between the input layer and hidden layer,
    # and between the hidden layer and output layer
    def initializeWeights(self) -> None:

        self.weightsInputHidden = [
            [random.uniform(-1.0, 1.0) for _ in range(self.inputNodes)]
            for _ in range(self.hiddenNodes)
        ]

        self.weightsHiddenOutput = [
            [random.uniform(-1.0, 1.0) for _ in range(self.hiddenNodes)]
            for _ in range(self.outputNodes)
        ]

    # This method creates bias values for the hidden layer and output layer
    def initializeBiases(self) -> None:

        self.biasHidden = [1.0 for _ in range(self.hiddenNodes)]
        self.biasOutput = [1.0 for _ in range(self.outputNodes)]

    # This method calculates the outputs of the hidden layer by applying
    # the dot product, adding the bias, and then applying the sigmoid function
    def calculateHiddenLayer(self, inputVector: list[float]) -> list[float]:

        if len(inputVector) != self.inputNodes:
            raise ValueError("Input vector size must match the number of input nodes")

        hiddenOutputs = []

        for hiddenIndex in range(self.hiddenNodes):
            weightedSum = ANNMath.dotProduct(inputVector, self.weightsInputHidden[hiddenIndex]) + self.biasHidden[hiddenIndex]
            hiddenOutput = ANNMath.sigmoid(weightedSum)
            hiddenOutputs.append(hiddenOutput)

        return hiddenOutputs

    # This method calculates the outputs of the output layer by applying
    # the dot product, adding the bias, and then applying the softmax function
    def calculateOutputLayer(self, hiddenOutputs: list[float]) -> list[float]:

        if len(hiddenOutputs) != self.hiddenNodes:
            raise ValueError("Hidden output size must match the number of hidden nodes")

        rawOutputs = []

        for outputIndex in range(self.outputNodes):
            weightedSum = ANNMath.dotProduct(hiddenOutputs, self.weightsHiddenOutput[outputIndex]) + self.biasOutput[outputIndex]
            rawOutputs.append(weightedSum)

        outputProbabilities = ANNMath.softmax(rawOutputs)
        return outputProbabilities

    # This method performs a forward pass through the network by first calculating the hidden layer outputs
    # and then calculating the output layer probabilities
    def forwardPass(self, inputVector: list[float]) -> list[float]:

        hiddenOutputs = self.calculateHiddenLayer(inputVector)
        outputProbabilities = self.calculateOutputLayer(hiddenOutputs)

        return outputProbabilities

    # This method performs a forward pass and returns the predicted class index
    def predict(self, inputVector: list[float]) -> int:

        outputProbabilities = self.forwardPass(inputVector)
        predictedClass = ANNMath.predictClass(outputProbabilities)

        return predictedClass

    # This method calculates the error at each output node by subtracting
    # the target value from the predicted probability
    def calculateOutputErrors(self, outputProbabilities: list[float], targetVector: list[float]) -> list[float]:

        if len(outputProbabilities) != self.outputNodes:
            raise ValueError("Output probabilities size must match the number of output nodes")

        if len(targetVector) != self.outputNodes:
            raise ValueError("Target vector size must match the number of output nodes")

        outputErrors = []

        for outputIndex in range(self.outputNodes):
            error = outputProbabilities[outputIndex] - targetVector[outputIndex]
            outputErrors.append(error)

        return outputErrors

    # This method calculates the error at each hidden node by using the
    # output errors, the hidden-to-output weights, and the derivative of sigmoid
    def calculateHiddenErrors(self, outputErrors: list[float], hiddenOutputs: list[float]) -> list[float]:

        if len(outputErrors) != self.outputNodes:
            raise ValueError("Output errors size must match the number of output nodes")

        if len(hiddenOutputs) != self.hiddenNodes:
            raise ValueError("Hidden outputs size must match the number of hidden nodes")

        hiddenErrors = []

        for hiddenIndex in range(self.hiddenNodes):
            weightedErrorSum = 0.0

            for outputIndex in range(self.outputNodes):
                weightedErrorSum += outputErrors[outputIndex] * self.weightsHiddenOutput[outputIndex][hiddenIndex]

            hiddenError = weightedErrorSum * hiddenOutputs[hiddenIndex] * (1 - hiddenOutputs[hiddenIndex])
            hiddenErrors.append(hiddenError)

        return hiddenErrors

    # This method updates the weights between the hidden layer and output layer
    def updateWeightsHiddenOutput(self, outputErrors: list[float], hiddenOutputs: list[float]) -> None:

        if len(outputErrors) != self.outputNodes:
            raise ValueError("Output errors size must match the number of output nodes")

        if len(hiddenOutputs) != self.hiddenNodes:
            raise ValueError("Hidden outputs size must match the number of hidden nodes")

        for outputIndex in range(self.outputNodes):
            for hiddenIndex in range(self.hiddenNodes):
                self.weightsHiddenOutput[outputIndex][hiddenIndex] -= (
                    self.learningRate * outputErrors[outputIndex] * hiddenOutputs[hiddenIndex]
                )

    # This method updates the biases for the output layer
    def updateBiasOutput(self, outputErrors: list[float]) -> None:

        if len(outputErrors) != self.outputNodes:
            raise ValueError("Output errors size must match the number of output nodes")

        for outputIndex in range(self.outputNodes):
            self.biasOutput[outputIndex] -= self.learningRate * outputErrors[outputIndex]

    # This method updates the weights between the input layer and hidden layer
    def updateWeightsInputHidden(self, hiddenErrors: list[float], inputVector: list[float]) -> None:

        if len(hiddenErrors) != self.hiddenNodes:
            raise ValueError("Hidden errors size must match the number of hidden nodes")

        if len(inputVector) != self.inputNodes:
            raise ValueError("Input vector size must match the number of input nodes")

        for hiddenIndex in range(self.hiddenNodes):
            for inputIndex in range(self.inputNodes):
                self.weightsInputHidden[hiddenIndex][inputIndex] -= (
                    self.learningRate * hiddenErrors[hiddenIndex] * inputVector[inputIndex]
                )

    # This method updates the biases for the hidden layer
    def updateBiasHidden(self, hiddenErrors: list[float]) -> None:

        if len(hiddenErrors) != self.hiddenNodes:
            raise ValueError("Hidden errors size must match the number of hidden nodes")

        for hiddenIndex in range(self.hiddenNodes):
            self.biasHidden[hiddenIndex] -= self.learningRate * hiddenErrors[hiddenIndex]

    # This method performs one full training step for a single input and target pair
    def trainSingleSample(self, inputVector: list[float], targetVector: list[float]) -> None:

        if len(inputVector) != self.inputNodes:
            raise ValueError("Input vector size must match the number of input nodes")

        if len(targetVector) != self.outputNodes:
            raise ValueError("Target vector size must match the number of output nodes")

        hiddenOutputs = self.calculateHiddenLayer(inputVector)
        outputProbabilities = self.calculateOutputLayer(hiddenOutputs)

        outputErrors = self.calculateOutputErrors(outputProbabilities, targetVector)
        hiddenErrors = self.calculateHiddenErrors(outputErrors, hiddenOutputs)

        self.updateWeightsHiddenOutput(outputErrors, hiddenOutputs)
        self.updateBiasOutput(outputErrors)
        self.updateWeightsInputHidden(hiddenErrors, inputVector)
        self.updateBiasHidden(hiddenErrors)

    # This method trains the ANN using the full training dataset over a number of epochs
    def train(self, trainingData: list[list[float]], epochs: int) -> None:

        if epochs <= 0:
            raise ValueError("Epochs must be greater than 0")

        for row in trainingData:
            if len(row) != self.inputNodes + 1:
                raise ValueError("Each training row must contain input values plus one class label")

        for _ in range(epochs):
            for row in trainingData:
                inputVector = row[:self.inputNodes]
                classLabel = int(row[self.inputNodes])
                targetVector = ANNMath.oneHotEncoding(classLabel, self.outputNodes)

                self.trainSingleSample(inputVector, targetVector)

    # This method evaluates the ANN on test data and returns the classification accuracy
    def evaluate(self, testData: list[list[float]]) -> float:

        if not testData:
            raise ValueError("Test data cannot be empty")

        for row in testData:
            if len(row) != self.inputNodes + 1:
                raise ValueError("Each test row must contain input values plus one class label")

        correctPredictions = 0

        for row in testData:
            inputVector = row[:self.inputNodes]
            actualClass = int(row[self.inputNodes])

            predictedClass = self.predict(inputVector)

            if predictedClass == actualClass:
                correctPredictions += 1

        accuracy = correctPredictions / len(testData)
        return accuracy