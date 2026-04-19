import pytest
from annMath import ANNMath

"""
This test suite verifies the correctness of the ANNMath class methods.
"""

# Test that dotProduct correctly computes the dot product of two vectors
def testDotProduct():
    result = ANNMath.dotProduct([1, 2, 3], [4, 5, 6])
    assert result == 32

# Test that dotProduct raises a ValueError when given vectors of different lengths
def testDotProductInvalidLength():
    with pytest.raises(ValueError):
        ANNMath.dotProduct([1, 2], [3])

# Test that sigmoid of 0 is 0.5
def testSigmoidZero():
    result = ANNMath.sigmoid(0)
    assert result == pytest.approx(0.5)

# Test that sigmoid of a large positive number is close to 1
def testSigmoidLargePositive():
    result = ANNMath.sigmoid(100)
    assert result > 0.999999

# Test that sigmoid of a large negative number is close to 0
def testSigmoidLargeNegative():
    result = ANNMath.sigmoid(-100)
    assert result < 1e-10

# Test that softmax outputs probabilities that sum to 1
def testSoftmaxSumsToOne():
    result = ANNMath.softmax([2.0, 1.0, 0.1])
    assert sum(result) == pytest.approx(1.0)

# Test that the largest input value corresponds to the largest probability in the softmax output
def testSoftmaxLargestValueGetsLargestProbability():
    result = ANNMath.softmax([1.0, 3.0, 2.0])
    assert result.index(max(result)) == 1

# Test that softmax raises a ValueError when given an empty list
def testSoftmaxEmptyInput():
    with pytest.raises(ValueError):
        ANNMath.softmax([])

# Test that oneHotEncoding returns a list with a single 1 at the correct index
def testOneHotEncoding():
    result = ANNMath.oneHotEncoding(3, 5)
    assert result == [0, 0, 0, 1, 0], "One-hot encoding did not produce expected output"

# Test that oneHotEncoding raises a ValueError when the class index is out of range
def testOneHotEncodeInvalidIndex():
    with pytest.raises(ValueError):
        ANNMath.oneHotEncoding(5, 3)

# Test that predictClass returns the index of the largest probability
def testPredictClass():
    probabilities = [0.1, 0.7, 0.2]
    predicted_class = ANNMath.predictClass(probabilities)
    assert predicted_class == 1, "Predict class did not return expected class"

# Test that predictClass raises a ValueError when given an empty list
def testPredictClassEmptyInput():
    with pytest.raises(ValueError):
        ANNMath.predictClass([])