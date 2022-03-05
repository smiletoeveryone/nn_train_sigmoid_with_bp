# in python to code a nn with backpropagation

steps:

i. Initialize Network.

ii. Forward Propagate.

iii.  Back Propagate Error.

vi. Train Network.



# neuron activation
activation = sum(weight_i * input_i) + bias

# sigmoid activation function
output = 1 / (1 + e^(-activation))

# calculate slope
derivative = output * (1.0 - output)

