"""
Cory Brynds

Instructions:
- Fill in the TODO sections marked with "DEFINE" or "INSERT CODE".
- Do NOT use automatic differentiation libraries (e.g., PyTorch autograd).
- Use numpy only for computations.
"""

from multiprocessing import parent_process
import pickle
from typing import ParamSpecArgs
import numpy as np
import os
import sys
from plot_utils import plot_all, plot_sweep_errors, plot_frobenius_norms

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        # Initialize momentum for weight and bias to 0
        self.momentum_W = np.zeros_like(W)
        self.momentum_b = np.zeros_like(b)
        
    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):
        batch_size = self.x.shape[0]
        grad_input = grad_output @ self.W
        
        # The gradient of the weight and bias and averaged over the batch size
        self.grad_W = grad_output.T @ self.x / batch_size + l2_penalty * self.W
        self.grad_b = grad_output.mean(axis=0)
        
        # Momentum update
        self.momentum_W = momentum * self.momentum_W - learning_rate * self.grad_W
        self.momentum_b = momentum * self.momentum_b - learning_rate * self.grad_b
        self.W += self.momentum_W
        self.b += self.momentum_b
        
        return grad_input

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
        self.x = x
        return np.maximum(0,x)

    def backward(
        self, 
        grad_output
    ):
        return grad_output * (self.x > 0)

# This is a class for a sigmoid layer followed by a cross entropy loss function, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def compute_ce_loss(self, y):
        # Output is clipped to avoid log(0)
        clipped = np.clip(self.output, 1e-12, 1 - 1e-12)
        self.ce_loss = -y * np.log(clipped) - (1 - y) * np.log(1 - clipped)
        return np.mean(self.ce_loss)
    
    def backward(self, grad_output):
        self.ce_loss_grad = self.output - grad_output
        return self.ce_loss_grad

class Net(object):

    def __init__(self, input_dims, hidden_units):
        # I chose to use He initialization: scale by sqrt(2 / input dim), initialize biases to 0
        self.fc1 = LinearTransform(
            np.random.randn(hidden_units, input_dims) * np.sqrt(2.0 / input_dims),
            np.zeros(hidden_units))
        self.relu = ReLU()
        self.fc2 = LinearTransform(
            np.random.randn(1, hidden_units) * np.sqrt(2.0 / hidden_units),
            np.zeros(1))
        self.sigmoid_cross_entropy = SigmoidCrossEntropy()

    def forward(self, x):
        x = self.fc1.forward(x)
        x = self.relu.forward(x)
        x = self.fc2.forward(x)
        x = self.sigmoid_cross_entropy.forward(x)
        return x
        
    def backward(self, x, y, learning_rate, momentum, l2_penalty):
        grad = self.sigmoid_cross_entropy.backward(y)
        grad = self.fc2.backward(grad, learning_rate, momentum, l2_penalty)
        grad = self.relu.backward(grad)
        grad = self.fc1.backward(grad, learning_rate, momentum, l2_penalty)
        
        return grad
    
    def train(self, x_batch, y_batch, learning_rate, momentum, l2_penalty):
        prediction = self.forward(x_batch)
        
        ce_loss = self.sigmoid_cross_entropy.compute_ce_loss(y_batch)
        l2_loss = (l2_penalty / 2) * (np.sum(self.fc1.W**2) + np.sum(self.fc2.W**2))
        total_loss = ce_loss + l2_loss
        
        self.backward(x_batch, y_batch, learning_rate, momentum, l2_penalty)
        return ce_loss, total_loss

    def evaluate(self, x, y, l2_penalty=0.0):
        predictions = self.forward(x)
        
        ce_loss = self.sigmoid_cross_entropy.compute_ce_loss(y)
        l2_loss = (l2_penalty / 2) * (np.sum(self.fc1.W**2) + np.sum(self.fc2.W**2))
        total_loss = ce_loss + l2_loss
        
        predicted_labels = (predictions >= 0.5).astype(float)
        error_rate = np.mean(predicted_labels != y)
        return ce_loss, total_loss, error_rate

# Auxillary function added to make sweeping parameters easier
def train_model(train_x, train_y, test_x, test_y, input_dims,
    hidden_units=15, learning_rate=0.01, momentum=0.8,
    l2_penalty=0.0001, batch_size=200, num_epochs=100, label=""):
    num_examples = train_x.shape[0]
    num_batches = num_examples // batch_size
    nnet = Net(input_dims, hidden_units)

    history = {
        "ce_loss": [],
        "total_loss": [],
        "train_error": [],
        "test_error": [],
        "w1_frobenius": [],
        "w2_frobenius": [],
    }

    for epoch in range(num_epochs):
        cumulative_ce_loss = 0.0
        cumulative_total_loss = 0.0

        for b in range(num_batches):
            start_index = b * batch_size
            end_index = start_index + batch_size
            x_batch = train_x[start_index:end_index]
            y_batch = train_y[start_index:end_index]

            ce_loss, total_loss = nnet.train(x_batch, y_batch, learning_rate, momentum, l2_penalty)
            cumulative_ce_loss += ce_loss
            cumulative_total_loss += total_loss
            print('\r  [{}] Epoch {:3d},  Avg.Loss = {:.4f}'.format(
                label, epoch + 1, cumulative_total_loss / (b + 1)), end='')

        train_ce_loss, train_total_loss, train_error = nnet.evaluate(train_x, train_y, l2_penalty)
        test_ce_loss, test_total_loss, test_error = nnet.evaluate(test_x, test_y, l2_penalty)

        print()
        print('Train Loss: {:.3f}  Train Error: {:.2f}%  '
                'Test Loss: {:.3f}  Test Error: {:.2f}%'.format(
                    train_total_loss, 100. * train_error,
                    test_total_loss, 100. * test_error))

        history["ce_loss"].append(train_ce_loss)
        history["total_loss"].append(train_total_loss)
        history["train_error"].append(train_error)
        history["test_error"].append(test_error)
        history["w1_frobenius"].append(np.linalg.norm(nnet.fc1.W, 'fro'))
        history["w2_frobenius"].append(np.linalg.norm(nnet.fc2.W))

    return history

# Auxillary function added to make sweeping parameters easier
def sweep_parameter(param_name, param_values, defaults, train_x, train_y,
                    test_x, test_y, input_dims, save_dir="results"):
    runs = []
    for val in param_values:
        config = defaults.copy()
        config[param_name] = val
        label = f"{param_name}={val}"
        print(f"\n{'='*60}")
        print(f"  Sweep: {label}")
        print(f"{'='*60}")

        history = train_model(
            train_x, train_y, test_x, test_y, input_dims,
            hidden_units=config["hidden_units"],
            learning_rate=config["learning_rate"],
            momentum=config["momentum"],
            l2_penalty=config["l2_penalty"],
            batch_size=config["batch_size"],
            num_epochs=config["num_epochs"],
            label=label,
        )

        epochs = list(range(1, config["num_epochs"] + 1))
        runs.append({
            "epochs": epochs,
            "error_rates": history["test_error"],
            "train_error_rates": history["train_error"],
            "w1_frobenius": history["w1_frobenius"],
            "w2_frobenius": history["w2_frobenius"],
            "label": label,
        })

    sweep_dir = os.path.join(save_dir, f"sweep_{param_name}")
    plot_sweep_errors(
        runs,
        title=f"Error Rates vs {param_name}",
        save_path=os.path.join(sweep_dir, "error_comparison.png"),
    )

    if param_name == "l2_penalty":
        plot_frobenius_norms(
            runs,
            title="Weight Norms vs L2 Penalty",
            save_path=os.path.join(sweep_dir, "frobenius_norms.png"),
        )

    return runs

# I kept the default training loop inside of main, followed by the function call to sweep the parameters
if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
        train_x = data['train_data']
        train_y = data['train_labels']
        test_x = data['test_data']
        test_y = data['test_labels']
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes') # load the data
        train_x = data[b'train_data']
        train_y = data[b'train_labels']
        test_x = data[b'test_data']
        test_y = data[b'test_labels']

    num_examples, input_dims = train_x.shape
    num_examples_test, _ = test_x.shape
    train_y = train_y.reshape(-1, 1).astype(np.float64)
    test_y = test_y.reshape(-1, 1).astype(np.float64)
    
    #Normalize the training data
    min_x = np.reshape(np.amin(train_x,axis=1),(num_examples,1));
    max_x = np.reshape(np.amax(train_x,axis=1),(num_examples,1));
    min_array = np.tile(min_x,input_dims);
    max_array = np.tile(max_x,input_dims);
    train_x = np.true_divide((train_x - min_array),(max_array - min_array));
  
    #Normalize the test data
    min_x = np.reshape(np.amin(test_x,axis=1),(num_examples_test,1));
    max_x = np.reshape(np.amax(test_x,axis=1),(num_examples_test,1));
    min_array = np.tile(min_x,input_dims);
    max_array = np.tile(max_x,input_dims);
    test_x = np.true_divide((test_x - min_array),(max_array - min_array));

    num_epochs = 100
    batch_size = 200
    num_batches = num_examples // batch_size
    hidden_units = 15
    learning_rate = 0.01
    momentum = 0.8
    l2_penalty = 0.0001

    history = {
        "ce_loss": [],
        "total_loss": [],
        "train_error": [],
        "test_error": []
    }

    nnet = Net(input_dims, hidden_units)

    for epoch in range(num_epochs):
        cumulative_ce_loss = 0.0
        cumulative_total_loss = 0.0

        for b in range(num_batches):
            start_index = b * batch_size
            end_index = start_index + batch_size
            x_batch = train_x[start_index:end_index]
            y_batch = train_y[start_index:end_index]

            ce_loss, total_loss = nnet.train(x_batch, y_batch, learning_rate, momentum, l2_penalty)
            cumulative_ce_loss += ce_loss
            cumulative_total_loss += total_loss
            print('\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                epoch + 1, b + 1, cumulative_total_loss / (b + 1)), end='')

        train_ce_loss, train_total_loss, train_error = nnet.evaluate(train_x, train_y, l2_penalty)
        test_ce_loss, test_total_loss, test_error = nnet.evaluate(test_x, test_y, l2_penalty)

        print()
        print('    Train Loss: {:.3f}    Train Error: {:.2f}%'.format(train_total_loss, 100. * train_error))
        print('    Test Loss:  {:.3f}    Test Error:  {:.2f}%'.format(test_total_loss, 100. * test_error))

        history["ce_loss"].append(train_ce_loss)
        history["total_loss"].append(train_total_loss)
        history["train_error"].append(train_error)
        history["test_error"].append(test_error)

    # Plot the default results
    plot_all(history, "Default ", "results")
    
    # Set all of the default parameters and sweep parameters for evaluating different hyperparameters
    defaults = {
        "hidden_units": 15,
        "learning_rate": 0.01,
        "batch_size": 200,
        "momentum": 0.8,
        "l2_penalty": 0.0001,
        "num_epochs": 100,
    }

    # Set the parameters to sweep
    sweeps = {
        "hidden_units": [5, 10, 15, 20, 25, 30],
        "learning_rate": [0.001, 0.01, 0.03, 0.05, 0.1],
        "momentum": [0, 0.5, 0.8, 0.9],
        "batch_size": [10, 50, 100, 200, 500],
        "l2_penalty": [0,0.01,0.001,0.0001]
    }

    # Evaluate network performance under all parameter combinations
    for param_name, param_values in sweeps.items():
        sweep_parameter(param_name, param_values, defaults,
                        train_x, train_y, test_x, test_y, input_dims)
    

