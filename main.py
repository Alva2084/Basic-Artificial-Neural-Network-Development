# Adolfo Alvarez Jr

import csv
import random
from simpleAnn import SimpleANN


"""
This file runs the ANN by loading the dataset, splitting it into training
and testing sets, training the network, and printing the results.
"""

# This method loads the dataset from a CSV file and converts each row
# into a list containing x value, y value, and class label
def loadData(filePath: str) -> list[list[float]]:

    data = []

    classMap = {
        "C1": 0,
        "C2": 1,
        "C3": 2
    }

    with open(filePath, "r", newline = "", encoding = "utf-8-sig") as file:
        reader = csv.reader(file)

        next(reader, None)

        for row in reader:
            if not row:
                continue

            try:
                xValue = float(row[0])
                yValue = float(row[1])
                classLabel = classMap[row[2].strip()]

                data.append([xValue, yValue, classLabel])

            except (ValueError, KeyError, IndexError):
                continue

    if not data:
        raise ValueError("Dataset is empty or could not be loaded")

    return data

# This method shuffles the dataset and splits it into training and testing sets
def splitData(data: list[list[float]], trainRatio: float = 0.8) -> tuple[list[list[float]], list[list[float]]]:

    if not data:
        raise ValueError("Data cannot be empty")

    if trainRatio <= 0 or trainRatio >= 1:
        raise ValueError("trainRatio must be between 0 and 1")

    shuffledData = data[:]
    random.shuffle(shuffledData)

    splitIndex = int(len(shuffledData) * trainRatio)

    trainingData = shuffledData[:splitIndex]
    testingData = shuffledData[splitIndex:]

    if not trainingData:
        raise ValueError("Training data is empty after split")

    if not testingData:
        raise ValueError("Testing data is empty after split")

    return trainingData, testingData

# This method runs the full ANN process using the dataset
def runDemo() -> None:

    print("--- SimpleANN Demo ---")

    random.seed(42)

    filePath = "data.csv"
    data = loadData(filePath)

    trainingData, testingData = splitData(data, 0.8)

    ann = SimpleANN(learningRate = 0.01)

    epochs = 1000
    ann.train(trainingData, epochs)

    accuracy = ann.evaluate(testingData)

    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(trainingData)}")
    print(f"Testing samples: {len(testingData)}")
    print(f"Epochs: {epochs}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Do the thing
def main() -> None:
    runDemo()


if __name__ == "__main__":
    main()