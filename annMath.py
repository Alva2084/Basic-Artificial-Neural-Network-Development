# Adolfo Alvarez Jr

import math
from typing import List

"""
This class contains static methods for performing mathematical operations commonly used in artificial neural networks (ANNs).
"""

class ANNMath:

    # This method takes two vectors (lists of numbers) and computes their dot product
    @staticmethod
    def dotProduct(vectorA: List[float], vectorB: List[float]) -> float:

        if len(vectorA) != len(vectorB):
            raise ValueError("Vectors must be of the same length")
        return sum(a * b for a, b in zip(vectorA, vectorB))

    # This method takes a raw output value and applies the sigmoid function to convert it into a value between 0 and 1
    @staticmethod
    def sigmoid(z: float) -> float:

        return 1 / (1 + math.exp(-z))

    # This method takes a list of raw output values and applies the softmax function to convert them into probabilities
    @staticmethod
    def softmax(values: List[float]) -> List[float]:

        if not values:
            raise ValueError("Softmax input list cannot be empty")

        maxValue = max(values)
        expValues = [math.exp(value - maxValue) for value in values]
        total = sum(expValues)

        return [value / total for value in expValues]

    # This method takes a class index and the total number of classes and returns a one-hot encoded list
    @staticmethod
    def oneHotEncoding(classIndex: int, totalClasses: int) -> List[int]:

        if classIndex < 0 or classIndex >= totalClasses:
            raise ValueError("Index must be within the range of the size")

        encoded = [0] * totalClasses
        encoded[classIndex] = 1
        return encoded

    # This method takes a list of probabilities and returns the index of the largest one
    @staticmethod
    def predictClass(probabilities: List[float]) -> int:

        if not probabilities:
            raise ValueError("Input probabilities list cannot be empty")
        return probabilities.index(max(probabilities))