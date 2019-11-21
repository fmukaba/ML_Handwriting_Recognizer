import numpy as np
import pandas as pd
from random import seed
from random import uniform

def createNetwork(num_inputs, num_hiddens, num_outputs):
    network = {}
    
    hiddens = num_inputs + 1
    hidden_list = []
    for r in range(num_hiddens):
        weights_list = []
        for i in range(hiddens): 
            w = getRandomWeight()
            weights_list.append(w)
        hidden_list.append(weights_list) 
        
    network["hidden"] = hidden_list 
    
    outputs = num_hiddens + 1
    output_list = [] 
    for r in range(num_outputs):
        weights_list = []
        for i in range(outputs): 
            w = getRandomWeight()
            weights_list.append(w)
        output_list.append(weights_list) 
    
    network["output"] = output_list 
    return network
    
def getRandomWeight():
    seed()
    
    return uniform(-0.05, 0.05)

def logistic_function(net_value):
    return 1 / (1 + np.e ** -net_value)


def net(weights, x_values):
    # Given two vectors: a weights vector and x_values vector
    
    # Function returns the dot product
    
    if len(weights) != len(x_values):
        raise Exception("In function net: Sizes of weights and x_values are different")

    return (pd.Series(weights) * pd.Series(x_values)).sum()

def propagate_forward(x_values, network):
   
    results = {}
    
    # propagate through hidden layer
    
    output_values = []
    hidden_layer = network["hidden"]
    
    # add input bias
    x_values.append(1)
    
    print(hidden_layer, "\n")
    print(x_values, "\n")
   
    for i in range(len(hidden_layer)):
        dot = net(hidden_layer[i],x_values)
        output = logistic_function(dot)
        output_values.append(output)
        
    # add bias to hidden layer's output
    
    results["hidden_r"] = output_values
    output_values.append(1)
    
    # propagate through output layer
    
    output_layer = network["output"]
    print(output_layer, "fer")
    output = []
    
    # print(output_layer, "\n")
    # print(output_values, "\n")
  
    for i in range(len(output_layer)):
        dot = net(output_layer[i], output_values)
        out = logistic_function(dot)
        output.append(out)
    
    # print(output)
    
    results["output_r"] = output_values
   
    return results

def propagate_backward(r_values, network):
    o_values = r_values["output_r"]
    h_values = r_values["hidden_r"]

    # get target label
    target = 1
    
    target_rep = []
    for i in range(len(o_values)):
        target_rep.append(0.01)
    # switch real target value on
    target_rep[target] = 0.99    
    
    errors_output = []
    for i in range(len(o_values)):
        o = o_values[i]
        err = o * (1 - o) * (target_rep[i] - o)
        errors_output.append(err)
    
    errors_hidden = []
   
    weights = network["output"]
    print(weights, "weight")
    print(errors_output, "err outp")
    
    for i in range(len(h_values)):
        h = h_values[i]
        err = h * (1 - h) * ((pd.Series(weights[i]) * pd.Series(errors_output)).sum())
        errors_hidden.append(err)

# update weights

        
network = createNetwork(2,2,2)

# print(network, "\n")
inputs = [20,20]

output = propagate_forward(inputs, network)
propagate_backward(output, network)

