# Author Francois Mukaba
# Course: Machine Learning
# Date : 11/23/2019
import numpy as np
import pandas as pd
from random import seed
from random import uniform


# Assign random weights to the network
def initialize_weights(num_inputs: int, num_hidden: int, num_outputs: int) -> {}:
    network_weights = {}

    # weights from input to hidden layer (including one bias)
    inputs = num_inputs + 1
    weights_hidden = []
    for r in range(num_hidden):
        node_weights = []
        for i in range(inputs):
            w = get_random_weight()
            node_weights.append(w)
        weights_hidden.append(node_weights)
    network_weights["hidden"] = weights_hidden

    # weights from hidden to output layer (including one bias)
    hidden = num_hidden + 1
    weights_output = []
    for r in range(num_outputs):
        node_weights = []
        for i in range(hidden):
            w = get_random_weight()
            node_weights.append(w)
        weights_output.append(node_weights)
    network_weights["output"] = weights_output

    return network_weights


def get_random_weight() -> int:
    seed()
    return uniform(-0.05, 0.05)


def logistic_function(net_value: int) -> int:
    return 1 / (1 + np.e ** -net_value)


# Compute dot product of two vectors
def net(x_values: [], y_values: []) -> int:
    if len(y_values) != len(x_values):
        raise Exception("In function net: Sizes of weights and x_values are different")
    return (pd.Series(y_values) * pd.Series(x_values)).sum()


# Returns a dictionary that contains input, hidden layer's output and final output
def propagate_forward(inputs: [], network: {}) -> {}:
    layers_outputs = {}

    # add input bias
    inputs.insert(0, 1)
    layers_outputs["input"] = inputs.copy()
    # propagate through hidden layer
    weights_hidden = network["hidden"]
    output_hidden = []
    for i in range(len(weights_hidden)):
        dot = net(weights_hidden[i], inputs)
        out = logistic_function(dot)
        output_hidden.append(out)
    layers_outputs["hidden"] = output_hidden.copy()

    # add bias to hidden layer's output
    output_hidden.insert(0, 1)

    # propagate through output layer
    weights_output = network["output"]
    output_values = []
    for i in range(len(weights_output)):
        dot = net(weights_output[i], output_hidden)
        out = logistic_function(dot)
        output_values.append(out)
    layers_outputs["output"] = output_values

    return layers_outputs


# Returns a dictionary that contains errors of the hidden and output layers
def propagate_backward(layers_output: {}, network: {}, target: int) -> {}:
    errors = {}
    output_values = layers_output["output"]

    # get target label
    target_label = target
    target_rep = []

    for i in range(len(output_values)):
        target_rep.append(0.01)
    # switch real target value on
    target_rep[target_label] = 0.99

    # get errors from output layer
    errors_output = []
    for i in range(len(output_values)):
        o = output_values[i]
        err = o * (1 - o) * (target_rep[i] - o)
        errors_output.append(err)
    errors["output"] = errors_output

    # get errors from hidden layer
    errors_hidden = []
    output_hidden = layers_output["hidden"]
    weights_output = network["output"]
    for i in range(len(weights_output[0])):
        if i != 0:  # not the bias
            weight_node = []
            for j in range(len(weights_output)):
                weight_node.append(weights_output[j][i])
            h = output_hidden[i - 1]
            err = h * (1 - h) * ((pd.Series(weight_node) * pd.Series(errors_output)).sum())
            errors_hidden.append(err)
    errors["hidden"] = errors_hidden

    return errors


# Updates network's weights
def update_weights(network: {}, errors: {}, layers_output: {}, learning_rate: float):
    # update weights from hidden to output layer
    weights_output = network["output"]
    errors_output = errors["output"]
    output_hidden = layers_output["hidden"]

    # adding back the bias
    output_hidden.insert(0, 1)

    for i in range(len(weights_output)):
        for j in range(len(weights_output[0])):
            weights_output[i][j] = weights_output[i][j] + (learning_rate * errors_output[i] * output_hidden[j])

    # update weights from input to hidden layer
    weights_hidden = network["hidden"]
    errors_hidden = errors["hidden"]
    input_hidden = layers_output["input"]

    for i in range(len(weights_hidden)):
        for j in range(len(weights_hidden[0])):
            weights_hidden[i][j] = weights_hidden[i][j] + (learning_rate * errors_hidden[i] * input_hidden[j])
