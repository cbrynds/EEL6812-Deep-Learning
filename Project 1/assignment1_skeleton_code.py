"""
INSERT YOUR NAME HERE

Instructions:
- Fill in the TODO sections marked with "DEFINE" or "INSERT CODE".
- Do NOT use automatic differentiation libraries (e.g., PyTorch autograd).
- Use numpy only for computations.
"""

import pickle
import numpy as np

# This is a class for a LinearTransform layer which takes an input 
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
	# DEFINE __init function

    def forward(self, x):
	# DEFINE forward function

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ): #retain only the relevant input arguments
	# DEFINE backward function
# ADD other operations in LinearTransform if needed

# This is a class for a ReLU layer max(x,0)
class ReLU(object):

    def forward(self, x):
	# DEFINE forward function

    def backward(
        self, 
        grad_output, 
        learning_rate=0.0, 
        momentum=0.0, 
        l2_penalty=0.0,
    ):#retain only the relevant input arguments
    # DEFINE backward function
# ADD other operations in ReLU if needed

# This is a class for a sigmoid layer followed by a cross entropy loss function, the reason 
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
	def forward(self, x):
		# DEFINE forward function
	def backward(
	    self, 
	    grad_output, 
		learning_rate=0.0,
		momentum=0.0,
		l2_penalty=0.0
	):#retain only the relevant input arguments
		# DEFINE backward function
# ADD other operations and data entries in SigmoidCrossEntropy if needed


# This is a class for the Neural Network
class Net(object):

    def __init__(self, input_dims, hidden_units):
    # INSERT CODE for initializing the network

    def train(
        self, 
        x_batch, 
        y_batch, 
        learning_rate, 
        momentum,
        l2_penalty,
    ):#retain only the relevant input arguments
	# INSERT CODE for training the network

    def evaluate(self, x, y):
	# INSERT CODE for testing the network
# ADD other operations and data entries if needed

if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
        train_x = data['train_data']
        train_y = data['train_labels']
        test_x = data['test_data']
        test_y = data['test_labels']
	else:
	    data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')
        train_x = data[b'train_data']
        train_y = data[b'train_labels']
        test_x = data[b'test_data']
        test_y = data[b'test_labels']
	
    num_examples, input_dims = train_x.shape
    
    

    #Normalize the data
    min_x = np.reshape(np.amin(train_x,axis=1),(num_examples,1));
    max_x = np.reshape(np.amax(train_x,axis=1),(num_examples,1));
    min_array = np.tile(min_x,input_dims);
    max_array = np.tile(max_x,input_dims);
    train_x = np.true_divide((train_x - min_array),(max_array - min_array));
  
  
	# INSERT YOUR CODE HERE
	# YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
	num_epochs = 50
	num_batches = 1000
    nnet = Net(input_dims, hidden_units)

    for epoch in xrange(num_epochs):

	# INSERT YOUR CODE FOR EACH EPOCH HERE

        for b in xrange(num_batches):
			total_loss = 0.0
			# INSERT YOUR CODE FOR EACH MINI_BATCH HERE
			# MAKE SURE TO UPDATE total_loss
            print(
                '\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(
                    epoch + 1,
                    b + 1,
                    total_loss,
                ),
                end='',
            )
		# INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
		# MAKE SURE TO COMPUTE train_loss, train_error, test_loss, test_error
        print()
        print('    Train Loss: {:.3f}    Train Error: {:.2f}%'.format(
            train_loss,
            100. * train_error,
        ))
        print('    Test Loss:  {:.3f}    Test Error:  {:.2f}%'.format(
            test_loss,
            100. * test_error,
        ))
