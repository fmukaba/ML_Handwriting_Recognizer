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

# input_df = pd.read_csv("data/training60000.csv", header=None)
# output_df = pd.read_csv("data/training60000_labels.csv", header=None)
# network = initialize_weights(INPUT_NUM, HIDDEN_NUM, OUTPUT_NUM)

networker = {'hidden': [[0.01, 0.03, 0.05], [0.02, 0.04, 0.01]], 'output': [[0.02, 0.01, 0.03], [0.01, 0.02, 0.04]]}
inputs = [0.02, 0.03]
values = propagate_forward(inputs, networker)
errors = propagate_backward(values, networker)
update_weights(networker, errors, values, LEARNING_RATE)
print(networker)
