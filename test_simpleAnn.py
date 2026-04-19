# Adolfo Alvarez Jr

from simpleAnn import SimpleANN

"""
This test suite contains unit tests for the SimpleANN class to verify that it initializes its properties correctly
and that the weight matrices and bias lists have the expected dimensions and values.
"""

# Test that the ANN stores the correct default number of input, hidden, and output nodes
def testDefaultNodeCounts():
    ann = SimpleANN()

    assert ann.inputNodes == 2
    assert ann.hiddenNodes == 5
    assert ann.outputNodes == 3

# Test that the ANN stores the learning rate correctly
def testLearningRate():
    ann = SimpleANN(learningRate = 0.25)

    assert ann.learningRate == 0.25

# Test that calculateHiddenLayer returns one value per hidden node
def testCalculateHiddenLayerOutputLength():
    ann = SimpleANN()

    inputVector = [0.5, -0.2]
    hiddenOutputs = ann.calculateHiddenLayer(inputVector)

    assert len(hiddenOutputs) == ann.hiddenNodes

# Test that each hidden layer output is between 0 and 1
def testCalculateHiddenLayerOutputRange():
    ann = SimpleANN()

    inputVector = [0.5, -0.2]
    hiddenOutputs = ann.calculateHiddenLayer(inputVector)

    for value in hiddenOutputs:
        assert 0.0 < value < 1.0

# Test that calculateHiddenLayer raises an error for invalid input size
def testCalculateHiddenLayerInvalidInputSize():
    ann = SimpleANN()

    try:
        ann.calculateHiddenLayer([1.0, 2.0, 3.0])
        assert False, "Expected ValueError for invalid input size"
    except ValueError:
        assert True

# Test that output layer returns correct number of outputs
def testCalculateOutputLayerOutputLength():
    ann = SimpleANN()

    hiddenOutputs = [0.1, 0.2, 0.3, 0.4, 0.5]
    outputProbabilities = ann.calculateOutputLayer(hiddenOutputs)

    assert len(outputProbabilities) == ann.outputNodes

# Test that output probabilities sum to 1
def testCalculateOutputLayerSumsToOne():
    ann = SimpleANN()

    hiddenOutputs = [0.1, 0.2, 0.3, 0.4, 0.5]
    outputProbabilities = ann.calculateOutputLayer(hiddenOutputs)

    total = sum(outputProbabilities)
    assert abs(total - 1.0) < 1e-9

# Test that output probabilities are between 0 and 1
def testCalculateOutputLayerProbabilityRange():
    ann = SimpleANN()

    hiddenOutputs = [0.1, 0.2, 0.3, 0.4, 0.5]
    outputProbabilities = ann.calculateOutputLayer(hiddenOutputs)

    for value in outputProbabilities:
        assert 0.0 <= value <= 1.0

# Test that calculateOutputLayer raises error for invalid input size
def testCalculateOutputLayerInvalidInputSize():
    ann = SimpleANN()

    try:
        ann.calculateOutputLayer([0.1, 0.2])
        assert False, "Expected ValueError for invalid hidden output size"
    except ValueError:
        assert True

# Test forwardPass output size
def testForwardPassOutputLength():
    ann = SimpleANN()

    inputVector = [0.5, -0.2]
    outputProbabilities = ann.forwardPass(inputVector)

    assert len(outputProbabilities) == ann.outputNodes

# Test forwardPass sums to 1
def testForwardPassSumsToOne():
    ann = SimpleANN()

    inputVector = [0.5, -0.2]
    outputProbabilities = ann.forwardPass(inputVector)

    total = sum(outputProbabilities)
    assert abs(total - 1.0) < 1e-9

# Test forwardPass probability range
def testForwardPassProbabilityRange():
    ann = SimpleANN()

    inputVector = [0.5, -0.2]
    outputProbabilities = ann.forwardPass(inputVector)

    for value in outputProbabilities:
        assert 0.0 <= value <= 1.0

# Test that predict returns an integer class index
def testPredictReturnsInteger():
    ann = SimpleANN()

    inputVector = [0.5, -0.2]
    predictedClass = ann.predict(inputVector)

    assert isinstance(predictedClass, int)

# Test that predict returns a valid class index
def testPredictReturnsValidClassIndex():
    ann = SimpleANN()

    inputVector = [0.5, -0.2]
    predictedClass = ann.predict(inputVector)

    assert 0 <= predictedClass < ann.outputNodes

# Test that calculateOutputErrors returns one error value for each output node
def testCalculateOutputErrorsLength():
    ann = SimpleANN()

    outputProbabilities = [0.1, 0.7, 0.2]
    targetVector = [0, 1, 0]

    outputErrors = ann.calculateOutputErrors(outputProbabilities, targetVector)

    assert len(outputErrors) == ann.outputNodes

# Test that calculateOutputErrors correctly subtracts the target values
def testCalculateOutputErrorsValues():
    ann = SimpleANN()

    outputProbabilities = [0.1, 0.7, 0.2]
    targetVector = [0, 1, 0]

    outputErrors = ann.calculateOutputErrors(outputProbabilities, targetVector)

    assert abs(outputErrors[0] - 0.1) < 1e-9
    assert abs(outputErrors[1] - (-0.3)) < 1e-9
    assert abs(outputErrors[2] - 0.2) < 1e-9

# Test that calculateOutputErrors raises a ValueError when output size is invalid
def testCalculateOutputErrorsInvalidOutputSize():
    ann = SimpleANN()

    outputProbabilities = [0.1, 0.7]
    targetVector = [0, 1, 0]

    try:
        ann.calculateOutputErrors(outputProbabilities, targetVector)
        assert False, "Expected ValueError for invalid output probabilities size"
    except ValueError:
        assert True

# Test that calculateOutputErrors raises a ValueError when target size is invalid
def testCalculateOutputErrorsInvalidTargetSize():
    ann = SimpleANN()

    outputProbabilities = [0.1, 0.7, 0.2]
    targetVector = [1, 0]

    try:
        ann.calculateOutputErrors(outputProbabilities, targetVector)
        assert False, "Expected ValueError for invalid target vector size"
    except ValueError:
        assert True

# Test that calculateHiddenErrors returns one error value for each hidden node
def testCalculateHiddenErrorsLength():
    ann = SimpleANN()

    outputErrors = [0.1, -0.3, 0.2]
    hiddenOutputs = [0.2, 0.4, 0.6, 0.8, 0.5]

    hiddenErrors = ann.calculateHiddenErrors(outputErrors, hiddenOutputs)

    assert len(hiddenErrors) == ann.hiddenNodes

# Test that calculateHiddenErrors returns numeric values
def testCalculateHiddenErrorsValueType():
    ann = SimpleANN()

    outputErrors = [0.1, -0.3, 0.2]
    hiddenOutputs = [0.2, 0.4, 0.6, 0.8, 0.5]

    hiddenErrors = ann.calculateHiddenErrors(outputErrors, hiddenOutputs)

    for value in hiddenErrors:
        assert isinstance(value, float)

# Test that calculateHiddenErrors raises a ValueError when output error size is invalid
def testCalculateHiddenErrorsInvalidOutputErrorSize():
    ann = SimpleANN()

    outputErrors = [0.1, -0.3]
    hiddenOutputs = [0.2, 0.4, 0.6, 0.8, 0.5]

    try:
        ann.calculateHiddenErrors(outputErrors, hiddenOutputs)
        assert False, "Expected ValueError for invalid output error size"
    except ValueError:
        assert True

# Test that calculateHiddenErrors raises a ValueError when hidden output size is invalid
def testCalculateHiddenErrorsInvalidHiddenOutputSize():
    ann = SimpleANN()

    outputErrors = [0.1, -0.3, 0.2]
    hiddenOutputs = [0.2, 0.4]

    try:
        ann.calculateHiddenErrors(outputErrors, hiddenOutputs)
        assert False, "Expected ValueError for invalid hidden output size"
    except ValueError:
        assert True

# Test that calculateHiddenErrors produces expected values with known weights
def testCalculateHiddenErrorsExpectedValues():
    ann = SimpleANN()

    ann.weightsHiddenOutput = [
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.2, 0.2, 0.2, 0.2]
    ]

    outputErrors = [0.1, -0.3, 0.2]
    hiddenOutputs = [0.2, 0.4, 0.6, 0.8, 0.5]

    hiddenErrors = ann.calculateHiddenErrors(outputErrors, hiddenOutputs)

    assert len(hiddenErrors) == 5
    assert hiddenErrors[0] != 0
    assert hiddenErrors[1] != 0
    assert hiddenErrors[2] != 0
    assert hiddenErrors[3] != 0
    assert hiddenErrors[4] != 0

# Test that updateWeightsHiddenOutput changes the weights
def testUpdateWeightsHiddenOutputChangesWeights():
    ann = SimpleANN(learningRate=0.1)

    ann.weightsHiddenOutput = [
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5, 0.5]
    ]

    originalWeights = [row[:] for row in ann.weightsHiddenOutput]

    outputErrors = [0.1, -0.3, 0.2]
    hiddenOutputs = [0.2, 0.4, 0.6, 0.8, 0.5]

    ann.updateWeightsHiddenOutput(outputErrors, hiddenOutputs)

    changed = False

    for i in range(ann.outputNodes):
        for j in range(ann.hiddenNodes):
            if ann.weightsHiddenOutput[i][j] != originalWeights[i][j]:
                changed = True

    assert changed, "Weights should have been updated"

# Test that updateWeightsHiddenOutput raises error for invalid sizes
def testUpdateWeightsHiddenOutputInvalidSizes():
    ann = SimpleANN()

    try:
        ann.updateWeightsHiddenOutput([0.1, 0.2], [0.1, 0.2, 0.3])
        assert False, "Expected ValueError for invalid sizes"
    except ValueError:
        assert True

# Test that updateBiasOutput changes the bias values
def testUpdateBiasOutputChangesValues():
    ann = SimpleANN(learningRate=0.1)

    ann.biasOutput = [0.5, 0.5, 0.5]
    originalBias = ann.biasOutput[:]

    outputErrors = [0.1, -0.3, 0.2]

    ann.updateBiasOutput(outputErrors)

    changed = False

    for i in range(len(ann.biasOutput)):
        if ann.biasOutput[i] != originalBias[i]:
            changed = True

    assert changed, "Output biases should have been updated"

# Test that updateBiasOutput raises error for invalid size
def testUpdateBiasOutputInvalidSize():
    ann = SimpleANN()

    try:
        ann.updateBiasOutput([0.1, 0.2])
        assert False, "Expected ValueError for invalid size"
    except ValueError:
        assert True

# Test that updateWeightsInputHidden changes the weights
def testUpdateWeightsInputHiddenChangesWeights():
    ann = SimpleANN(learningRate=0.1)

    ann.weightsInputHidden = [
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5]
    ]

    originalWeights = [row[:] for row in ann.weightsInputHidden]

    hiddenErrors = [0.1, -0.2, 0.3, -0.4, 0.5]
    inputVector = [0.2, 0.8]

    ann.updateWeightsInputHidden(hiddenErrors, inputVector)

    changed = False

    for hiddenIndex in range(ann.hiddenNodes):
        for inputIndex in range(ann.inputNodes):
            if ann.weightsInputHidden[hiddenIndex][inputIndex] != originalWeights[hiddenIndex][inputIndex]:
                changed = True

    assert changed, "Input-to-hidden weights should have been updated"

# Test that updateWeightsInputHidden raises error for invalid hidden error size
def testUpdateWeightsInputHiddenInvalidHiddenErrorSize():
    ann = SimpleANN()

    try:
        ann.updateWeightsInputHidden([0.1, 0.2], [0.5, -0.2])
        assert False, "Expected ValueError for invalid hidden error size"
    except ValueError:
        assert True

# Test that updateWeightsInputHidden raises error for invalid input vector size
def testUpdateWeightsInputHiddenInvalidInputSize():
    ann = SimpleANN()

    try:
        ann.updateWeightsInputHidden([0.1, 0.2, 0.3, 0.4, 0.5], [0.5, -0.2, 0.7])
        assert False, "Expected ValueError for invalid input vector size"
    except ValueError:
        assert True

# Test that updateBiasHidden changes the hidden bias values
def testUpdateBiasHiddenChangesValues():
    ann = SimpleANN(learningRate=0.1)

    ann.biasHidden = [0.5, 0.5, 0.5, 0.5, 0.5]
    originalBias = ann.biasHidden[:]

    hiddenErrors = [0.1, -0.2, 0.3, -0.4, 0.5]

    ann.updateBiasHidden(hiddenErrors)

    changed = False

    for i in range(len(ann.biasHidden)):
        if ann.biasHidden[i] != originalBias[i]:
            changed = True

    assert changed, "Hidden biases should have been updated"

# Test that updateBiasHidden raises error for invalid size
def testUpdateBiasHiddenInvalidSize():
    ann = SimpleANN()

    try:
        ann.updateBiasHidden([0.1, 0.2])
        assert False, "Expected ValueError for invalid size"
    except ValueError:
        assert True

# Test that trainSingleSample changes at least one hidden-to-output weight
def testTrainSingleSampleChangesHiddenOutputWeights():
    ann = SimpleANN(learningRate=0.1)

    originalWeights = [row[:] for row in ann.weightsHiddenOutput]

    inputVector = [0.5, -0.2]
    targetVector = [0, 1, 0]

    ann.trainSingleSample(inputVector, targetVector)

    changed = False

    for outputIndex in range(ann.outputNodes):
        for hiddenIndex in range(ann.hiddenNodes):
            if ann.weightsHiddenOutput[outputIndex][hiddenIndex] != originalWeights[outputIndex][hiddenIndex]:
                changed = True

    assert changed, "Hidden-to-output weights should change after training"

# Test that trainSingleSample changes at least one input-to-hidden weight
def testTrainSingleSampleChangesInputHiddenWeights():
    ann = SimpleANN(learningRate=0.1)

    originalWeights = [row[:] for row in ann.weightsInputHidden]

    inputVector = [0.5, -0.2]
    targetVector = [0, 1, 0]

    ann.trainSingleSample(inputVector, targetVector)

    changed = False

    for hiddenIndex in range(ann.hiddenNodes):
        for inputIndex in range(ann.inputNodes):
            if ann.weightsInputHidden[hiddenIndex][inputIndex] != originalWeights[hiddenIndex][inputIndex]:
                changed = True

    assert changed, "Input-to-hidden weights should change after training"

# Test that trainSingleSample changes the output biases
def testTrainSingleSampleChangesOutputBiases():
    ann = SimpleANN(learningRate=0.1)

    originalBiases = ann.biasOutput[:]

    inputVector = [0.5, -0.2]
    targetVector = [0, 1, 0]

    ann.trainSingleSample(inputVector, targetVector)

    changed = False

    for index in range(ann.outputNodes):
        if ann.biasOutput[index] != originalBiases[index]:
            changed = True

    assert changed, "Output biases should change after training"

# Test that trainSingleSample changes the hidden biases
def testTrainSingleSampleChangesHiddenBiases():
    ann = SimpleANN(learningRate=0.1)

    originalBiases = ann.biasHidden[:]

    inputVector = [0.5, -0.2]
    targetVector = [0, 1, 0]

    ann.trainSingleSample(inputVector, targetVector)

    changed = False

    for index in range(ann.hiddenNodes):
        if ann.biasHidden[index] != originalBiases[index]:
            changed = True

    assert changed, "Hidden biases should change after training"

# Test that trainSingleSample raises error for invalid input size
def testTrainSingleSampleInvalidInputSize():
    ann = SimpleANN()

    try:
        ann.trainSingleSample([0.1, 0.2, 0.3], [0, 1, 0])
        assert False, "Expected ValueError for invalid input size"
    except ValueError:
        assert True

# Test that trainSingleSample raises error for invalid target size
def testTrainSingleSampleInvalidTargetSize():
    ann = SimpleANN()

    try:
        ann.trainSingleSample([0.1, 0.2], [1, 0])
        assert False, "Expected ValueError for invalid target size"
    except ValueError:
        assert True

# Test that train changes at least one hidden-to-output weight after training on data
def testTrainChangesHiddenOutputWeights():
    ann = SimpleANN(learningRate=0.1)

    originalWeights = [row[:] for row in ann.weightsHiddenOutput]

    trainingData = [
        [0.1, 0.2, 0],
        [0.5, -0.3, 1],
        [-0.4, 0.7, 2]
    ]

    ann.train(trainingData, 5)

    changed = False

    for outputIndex in range(ann.outputNodes):
        for hiddenIndex in range(ann.hiddenNodes):
            if ann.weightsHiddenOutput[outputIndex][hiddenIndex] != originalWeights[outputIndex][hiddenIndex]:
                changed = True

    assert changed, "Hidden-to-output weights should change after training"

# Test that train changes at least one input-to-hidden weight after training on data
def testTrainChangesInputHiddenWeights():
    ann = SimpleANN(learningRate=0.1)

    originalWeights = [row[:] for row in ann.weightsInputHidden]

    trainingData = [
        [0.1, 0.2, 0],
        [0.5, -0.3, 1],
        [-0.4, 0.7, 2]
    ]

    ann.train(trainingData, 5)

    changed = False

    for hiddenIndex in range(ann.hiddenNodes):
        for inputIndex in range(ann.inputNodes):
            if ann.weightsInputHidden[hiddenIndex][inputIndex] != originalWeights[hiddenIndex][inputIndex]:
                changed = True

    assert changed, "Input-to-hidden weights should change after training"

# Test that train raises a ValueError when epochs is invalid
def testTrainInvalidEpochs():
    ann = SimpleANN()

    trainingData = [
        [0.1, 0.2, 0]
    ]

    try:
        ann.train(trainingData, 0)
        assert False, "Expected ValueError for invalid epochs"
    except ValueError:
        assert True

# Test that train raises a ValueError when a row has the wrong size
def testTrainInvalidRowSize():
    ann = SimpleANN()

    trainingData = [
        [0.1, 0.2],
        [0.5, -0.3, 1]
    ]

    try:
        ann.train(trainingData, 5)
        assert False, "Expected ValueError for invalid row size"
    except ValueError:
        assert True

# Test that evaluate returns a float
def testEvaluateReturnsFloat():
    ann = SimpleANN()

    testData = [
        [0.1, 0.2, 0],
        [0.5, -0.3, 1],
        [-0.4, 0.7, 2]
    ]

    accuracy = ann.evaluate(testData)

    assert isinstance(accuracy, float)

# Test that evaluate returns a value between 0 and 1
def testEvaluateReturnsValidAccuracyRange():
    ann = SimpleANN()

    testData = [
        [0.1, 0.2, 0],
        [0.5, -0.3, 1],
        [-0.4, 0.7, 2]
    ]

    accuracy = ann.evaluate(testData)

    assert 0.0 <= accuracy <= 1.0

# Test that evaluate raises a ValueError when test data is empty
def testEvaluateEmptyData():
    ann = SimpleANN()

    try:
        ann.evaluate([])
        assert False, "Expected ValueError for empty test data"
    except ValueError:
        assert True

# Test that evaluate raises a ValueError when a test row has the wrong size
def testEvaluateInvalidRowSize():
    ann = SimpleANN()

    testData = [
        [0.1, 0.2],
        [0.5, -0.3, 1]
    ]

    try:
        ann.evaluate(testData)
        assert False, "Expected ValueError for invalid row size"
    except ValueError:
        assert True