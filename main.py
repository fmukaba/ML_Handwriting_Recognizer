# Author Francois Mukaba
# Course: Machine Learning
# Date : 11/23/2019
import pandas as pd
from back_propagation import initialize_weights, propagate_forward, propagate_backward, update_weights

# Constants
INPUT_NUM = 784
HIDDEN_NUM = 16
OUTPUT_NUM = 10

LEARNING_RATE = 0.05
EPOCH = 75
TRAINING_INSTANCES = 5000
TESTING_INSTANCES = 10000

training_input_df = pd.read_csv("data/training60000.csv", header=None)
training_output_df = pd.read_csv("data/training60000_labels.csv", header=None)
network_weights = initialize_weights(INPUT_NUM, HIDDEN_NUM, OUTPUT_NUM)


# Train the model
for e in range(EPOCH):
    for instance in range(TRAINING_INSTANCES):
        inputs = list(training_input_df.iloc[instance])
        outputs = list(training_output_df.iloc[instance])
        target_label = outputs[0]
        values = propagate_forward(inputs, network_weights)
        errors = propagate_backward(values, network_weights, target_label)
        update_weights(network_weights, errors, values, LEARNING_RATE)


# Test the model
testing_input_df = pd.read_csv("data/testing10000.csv", header=None)
testing_output_df = pd.read_csv("data/testing10000_labels.csv", header=None)

correct_classification = 0
incorrect_classification = 0
for instance in range(TESTING_INSTANCES):
    inputs = list(testing_input_df.iloc[instance])
    real_outputs = list(testing_output_df.iloc[instance])
    target_label = real_outputs[0]
    model_outputs = propagate_forward(inputs, network_weights)
    # If target output equals output
    if target_label == model_outputs["output"].index(max(model_outputs["output"])):
        correct_classification += 1
    else:
        incorrect_classification += 1

# Print test results
print("==== Results =====")
print("Network properties: Input: {0}, Hidden: {1}, Output: {2}".format(INPUT_NUM, HIDDEN_NUM, OUTPUT_NUM))
print("Learning rate: {0}, Epoch: {1}".format(LEARNING_RATE, EPOCH))
print("Correct classification = ", correct_classification)
print("Incorrect classification = ", TESTING_INSTANCES - correct_classification)
print("Accuracy = ", (correct_classification / TESTING_INSTANCES) * 100, "%")
