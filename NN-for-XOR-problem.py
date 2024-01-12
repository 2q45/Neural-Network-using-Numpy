import numpy as np

class Layer:
  def __init__(self,):
    self.input = None
    self.output = None

  def ForwardPropagation(self, input):
     # forward takes input and gives output
    pass
  def BackwardPropagation(self, previous_input_gradient, learning_rate):
    # previous input gradient is the derivative of
    # the cost/error with respect to the previous layer input
    # returns the new trainable parameter (update) and new input gradient

    pass

class DenseLayer(Layer):
  def __init__(self, input_nuerons, output_nuerons):
    self.weights = np.random.randn(output_nuerons, input_nuerons) # compute a random matrix of weights with rows output nuerons and collums input nuerons
    self.biases = np.random.randn(output_nuerons, 1)

  def ForwardPropagation(self, input):
    self.input = input
    return np.dot(self.weights, self.input) + self.biases

  def BackwardPropagation(self, previous_input_gradient, learning_rate):
    weights_gradient = np.dot(previous_input_gradient, self.input.T)
    self.biases -= learning_rate * previous_input_gradient
    self.weights -= learning_rate * weights_gradient
    return np.dot(self.weights.T,previous_input_gradient)


class Activation(Layer):
  def __init__(self, activation, activation_derivative):
    self.activation = activation
    self.activation_derivative = activation_derivative

  def ForwardPropagation(self, input):
    self.input = input
    return self.activation(self.input)

  def BackwardPropagation(self, previous_input_gradient, learning_rate):
    return np.multiply(previous_input_gradient, self.activation(self.input))


class tanh(Activation):
  def __init__(self):
    tanh = lambda x: np.tanh(x) # definings tanh as a function that takes input x
    tanh_derivative = lambda x: 1 - np.tanh(x)**2 # defining the derivative of tanh
    super().__init__(tanh, tanh_derivative) # call the super initialisation function passing both as arguments

def mse(y_true, y_pred):
  return np.mean(np.power(y_true-y_pred,2))

def mse_derivative(y_true, y_pred):
  return 2 * (y_pred-y_true)/np.size(y_true)



X = np.reshape([[0,0],[1,0],[0,1],[1,1]], (4,2,1)) # reshape to 4,2,1
Y = np.reshape([[0],[1],[1],[0]], (4,1,1))

network = [
    DenseLayer(2,69),
    tanh(),
    DenseLayer(69,420),
    tanh(),
    DenseLayer(420,1),
    tanh(),
    DenseLayer(1,1)
]

epochs = 3000
learning_rate = 0.001

for epoch in range(epochs):
  error = 0

  for x,y in zip(X,Y):
    output = x

    for layer in network:
      output = layer.ForwardPropagation(output)

    error += mse(y,output)

    grad = mse_derivative(y, output)

    for layer in reversed(network):
      grad = layer.BackwardPropagation(grad, learning_rate)

    error /= len(x)
    print(f" Epoch Number: {epoch+1},Loss/Error = {error}")
