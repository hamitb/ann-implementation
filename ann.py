import numpy as np
from layers import *

np.random.seed(499)

class ANN(object):
    """
    This class implements a modular fully-connected neural network with an 
    arbitrary number of hidden layers, ReLU nonlinearities, and a L2 loss
    function. For a network with L layers, the architecture will be

    {affine - relu} x (L - 1) - affine - loss

    where the {...} block is repeated L - 1 times.
    """

    def __init__(self, hidden_dims, input_dim, weight_scale=1e-2, dtype=np.float32):
        """
        In this part of the code, all parameters of the network should be
        initialized and stored in the self.params dictionary. For a layer "l", you
        should store the weights as "Wl" and biases as "bl". For example, for the 
        first layer, "W0" and "b0" should be initialized and stored. These
        parameters will be used in training and prediction phases. Weights should
        be initialized with random numbers frome a normal distribution having zero
        mean and standard deviation equal to weight_scale argument. Biases should
        be initialized to zero.

        Inputs:
        hidden_dims: dimensions of hidden layers, a list object.
        input_dim: dimension of data vectors
        weight_scale: scale of initial random weights
        dtype: type of the parameters
        """
        self.dtype = dtype
        self.params = {}
        self.layers_count = 1 + len(hidden_dims)
        self.dims = [input_dim] + hidden_dims + [1]

        # Initialize parameters
        # Dimentions input_dim, ...(hidden_dims), loss
        for i in range(len(self.dims)-1):
            dim, nextdim = self.dims[i], self.dims[i+1]
            Wi = weight_scale * np.random.randn(dim, nextdim)
            bi = np.zeros(nextdim)
            self.params["W{}".format(i)] = Wi
            self.params["b{}".format(i)] = bi
        
        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        This function is used for computing the loss and gradient for the network.

        Inputs:
        X: input data, an array of shape (N, d)
        y: labels, an array of shape (N,). y[i] is the label of datum X[i].

        Outputs:
        If y is None, then run a test-time forward pass of the model and return:
        - predictions: an array of shape (N,) giving target predictions, where
          preds[i] is the regression output for X[i].
        Else return:
        - loss: data loss computed using L2_loss
        - grads: a dictionary of gradients where the keys are the parameters of
          the network.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        preds = None

        """
        TODO: Implement the forward pass for the network using forward functions
        implemented in layers.py. Compute the predictions of the target variable
        for each datum and store them in the preds variable. Hint: You can use a
        dictionary like self.params to store intermediate results.
        """
        prev_activations= X
        self.params["a-1"] = X

        for i in range(len(self.dims)-1):
            # Get weight and bias for current layer
            Wi, bi = self.params["W{}".format(i)], self.params["b{}".format(i)]
            
            # Calculate o_i and a_i
            affine_out, _ = affine_forward(prev_activations, Wi, bi)
            relu_out, _ = relu_forward(affine_out)
            
            # Save calculated values
            self.params["o{}".format(i)] = affine_out
            self.params["a{}".format(i)] = relu_out

            # Prepare for next iteration
            prev_activations = relu_out
        
        # If test mode return early
        last_layer_index = self.layers_count-1
        last_layer_output = self.params["o{}".format(last_layer_index)]
        preds = last_layer_output

        if mode == 'test':
        	return preds

        loss, grads = 0.0, {}

        """
        TODO: Implement the backward pass for the network using backward functions
        implemented in layers.py. Compute data loss using L2_loss and gradients
        for the parameters. Store the gradient for parameter self.params[p] in
        grads[p].
        """
        
        loss, dout = L2_loss(last_layer_output, y)
        x = self.params["a{}".format(last_layer_index-1)]
        bi = self.params["b{}".format(last_layer_index)]
        Wi = self.params["W{}".format(last_layer_index)]
        cache = (x, Wi, bi)
        dout, dw, db = affine_backward(dout, cache)

        grads["W{}".format(last_layer_index)] = dw
        grads["b{}".format(last_layer_index)] = db

        for i in reversed(range(self.layers_count-1)):
            x = self.params["a{}".format(i-1)]
            bi = self.params["b{}".format(i)]
            Wi = self.params["W{}".format(i)]
            cache = (x, Wi, bi)
            
            relu_input = self.params["o{}".format(i)]
            
            d_relu = relu_backward(dout, relu_input)
            
            dout, dw, db = affine_backward(d_relu, cache)
            
            grads["W{}".format(i)] = dw
            grads["b{}".format(i)] = db

        return loss, grads
    def train_validate(self, X_t, y_t, X_v, y_v, maxEpochs=100, learning_rate=1e-4):
        """
        Train the network using gradient descent algorithm.

        Inputs:
        X_t: training data, an array of shape (N, d)
        y_t: training labels, an array of shape (N,). y_t[i] is the label of datum X_t[i].
        X_v: validation data, an array of shape (M, d)
        y_v: validation labels, an array of shape (M,). y_v[i] is the label of datum X_v[i].
        maxEpochs: maximum number of epochs that will be spent for training if not
                    stopped early.
        learning_rate: hyperparameter used in traditional gradient descent algorithm

        Outputs:
        loss_train: Loss history for training set containing loss for each epoch
        loss_valid: Loss history for validation set containing loss for each epoch
        """
        loss_train = []
        loss_valid = []
        
        for i in range(maxEpochs):
            cur_loss_train, grads = self.loss(X_t, y_t)
            cur_loss_valid, _ = self.loss(X_v, y_v)
            
            print "Training loss: {}\tValidation loss: {}\tEpoch {}/{}"\
                   .format(cur_loss_train, cur_loss_valid, i, maxEpochs)
            loss_train.append(cur_loss_train)
            loss_valid.append(cur_loss_valid)

            # Update the weights and biases using gradient descent
            for param, grad in grads.items():
                self.params[param] += -learning_rate * grad
            
        return loss_train, loss_valid
    def predict(self, X):
        """
        Predict the target variable using the trained network.

        Inputs:
        X: test data, an array of shape (N, d)

        Outputs:
        preds: an array of shape (N,) giving target predictions, where
               preds[i] is the regression output for X[i].
        """
        return self.loss(X,None)
