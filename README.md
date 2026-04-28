# Basic-Artificial-Neural-Network-Development
This project is creating a simple Artificial Neural Network from scratch to classify a two-dimensional dataset.

## Features
- Programming Language: Python
- Scratch implementation of Artificial Neural Network (ANN)
- Multi-class classification with 3 output classes
- Forward propagation using sigmoid (hidden layer) and sofmax (output layer)
- Backpropagation for updating weights and biases
- One-hot encoding for target representation
- Training and testing dataset split with accuracy evaluation
- Configurable hyperparameters (learning rate and epochs)
- No external machine libraries usesage

## Project Structure
  Basic Artificial Neural Network Development/
  - README.md
  - annMath.py
  - data.csv
  - main.py
  - simpleAnn.py
  - test_annMath.py
  - test_simpleAnn.py
## Requirements
- Python 3 or higher
- pytest:
  - pip install pytest (command line)
- Terminal or IDE
## How to Run
To Run the Tests:
- Bash the following command
  - pytest [test_filename]

To Run the Program:
- Bash the following command
  - python main.py

or

- Run main.py on your IDE
## Usage Notes
- The ANN uses random initialization of weights, so results may vary slightly between runs
- A fixed random seed is used in main.py to improve consistency of results
- The dataset contains overlapping classes, so perfect accuracy is not expected
- Accuracy improves with:
  - More epochs (longer training)
  - Smaller learning rates (more stable learning)
- Increasing epochs will increase runtime
- Hyperparameters such as lrarningRate and epochs can be adjusted in main.py to experiment with performance

## Resources


### Understanding One-Hot Encoding
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
- https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/
- https://www.educative.io/blog/one-hot-encoding

### Understanding Epoch
- https://deepchecks.com/glossary/epoch-in-machine-learning/
- https://nebius.com/blog/posts/epoch-in-machine-learning
- https://www.sihttps://www.simplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learningmplilearn.com/tutorials/machine-learning-tutorial/what-is-epoch-in-machine-learning

### Use of AI
Used the help of chatGPT to think of test cases to use for TDD which is where I came across one-hot encoding and epoch. This helped me crate different test cases to pay attention to when developing the project and with the help of coPilot, it gave suggestions close to what I was looking for.
