import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from plotting_utils import (
    plot_total_loss,
    plot_training_accuracy,
    plot_testing_accuracy,
)

class Net(nn.Module):
    def __init__(self, 
        add_batch_normalization=False, add_fc_layer=False, 
        reduce_hidden_nodes=False, halve_num_filters=False, 
        use_mean_pooling=False, add_dropout_layer=False):
        super(Net, self).__init__()
        
        self.add_batch_normalization = add_batch_normalization
        self.add_fc_layer = add_fc_layer
        self.reduce_hidden_nodes = reduce_hidden_nodes
        self.halve_num_filters = halve_num_filters
        self.use_mean_pooling = use_mean_pooling
        self.add_dropout_layer = add_dropout_layer
        
        # For ablation task 4
        block1_filter_count = 16 if halve_num_filters else 32
        block2_filter_count = 32 if halve_num_filters else 64
        
        # For ablation task 3
        fc2_input_size = 256 if reduce_hidden_nodes else 512

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=block1_filter_count, kernel_size=3, padding=1, stride=1) # conv1: 32 channels, 3x3 kernel, padding 1, stride 1
        self.conv2 = nn.Conv2d(in_channels=block1_filter_count, out_channels=block1_filter_count, kernel_size=3, padding=1, stride=1) # conv2: 32 channels, 3x3 kernel, padding 1, stride 1
        self.conv3 = nn.Conv2d(in_channels=block1_filter_count, out_channels=block2_filter_count, kernel_size=3, padding=1, stride=1) # conv3: 64 channels, 3x3 kernel, padding 1, stride 1
        self.conv4 = nn.Conv2d(in_channels=block2_filter_count, out_channels=block2_filter_count, kernel_size=3, padding=1, stride=1) # conv4: 64 channels, 3x3 kernel, padding 1, stride 1
        self.fc1 = nn.Linear(block2_filter_count * 8 * 8, 512) # fc1: 512 hidden nodes
        self.fc2 = nn.Linear(fc2_input_size, 10) # fc2: 10 nodes
        
        # For ablation task 5
        if use_mean_pooling:
            self.mp1 = nn.AvgPool2d(kernel_size=2, stride=2) # mp1: 2x2 kernel, stride 2
            self.mp2 = nn.AvgPool2d(kernel_size=2, stride=2) # mp2: 2x2 kernel, stride 2
        else:
            self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2) # mp1: 2x2 kernel, stride 2
            self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2) # mp2: 2x2 kernel, stride 2
        
        # Optional layers for ablation study
        self.bn = nn.BatchNorm1d(512)               # For Task 1
        self.fc_new = nn.Linear(512,fc2_input_size) # For Tasks 2,3
        self.dropout = nn.Dropout(p=0.5)            # For Task 6

    # Define the forward function
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.mp1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.mp2(x)
        
        # Flatten for FC layer
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        
        # Apply optional layers
        if self.add_batch_normalization:
            x = self.bn(x)
            
        if self.add_dropout_layer:
            x = self.dropout(x)
            
        # fc1 activation
        x = F.relu(x)

        if self.add_fc_layer:
            x = F.relu(self.fc_new(x))
        # No softmax, output logits directly and use nn.CrossEntropyLoss()
        x = self.fc2(x)

        return x

def eval_net(net, dataloader, criterion, device):
    # Evaluate model and return average loss + error rate
    net.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            _, predicted = torch.max(outputs.data, 1)
            total += batch_size
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / total
    acc_rate = correct / total
    return avg_loss, acc_rate

def train_and_evaluate_model(
    batch_size, learning_rate, max_epochs, 
    add_batch_normalization=False, reduce_hidden_nodes=False, 
    halve_num_filters=False, use_mean_pooling=False, 
    add_fc_layer=False, add_dropout_layer=False):
    
    # Load data
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #torchvision.transforms.Normalize(mean, std)

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    print('Building model...')
    
    # I'm working on MacOS, so I added a check for an MPS backend
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = torch.device(
        "mps" if has_mps
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f'Using device: {device}')

    # Initialize model, loss function, and optimizer
    net = Net(add_batch_normalization=add_batch_normalization, add_fc_layer=add_fc_layer,
        reduce_hidden_nodes=reduce_hidden_nodes, halve_num_filters=halve_num_filters,
        use_mean_pooling=use_mean_pooling, add_dropout_layer=add_dropout_layer).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Lists for baseline vs modified comparisons.
    total_loss_list = []
    training_accuracy_list = []
    testing_accuracy_list = []

    print('Start training...')
    for epoch in range(max_epochs):
        net.train()
        running_loss = 0.0
        epoch_loss_sum = 0.0
        epoch_sample_count = 0
        for i, data in enumerate(trainloader, 0):
            # Forward pass
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward() # Backpropagation
            optimizer.step() # mini batch gradient descent-based updates
            
            running_loss += loss.item()
            batch_size = labels.size(0)
            epoch_loss_sum += loss.item() * batch_size
            epoch_sample_count += batch_size

            # Print statistics
            if i % 100 == 99:    # print every 100 mini-batches
                print(f'Epoch [{epoch+1}/{max_epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0

        print(f'Finish training Epoch {epoch+1}, start evaluating...')
        train_loss, train_accuracy = eval_net(net, trainloader, criterion, device)
        test_loss, test_accuracy = eval_net(net, testloader, criterion, device)
        epoch_total_loss = epoch_loss_sum / epoch_sample_count if epoch_sample_count > 0 else 0.0

        total_loss_list.append(epoch_total_loss)
        training_accuracy_list.append(train_accuracy)
        testing_accuracy_list.append(test_accuracy)

        print('Epoch: %d total_loss: %.5f train_loss: %.5f train_accuracy: %.5f test_loss: %.5f test_accuracy: %.5f' %
              (epoch+1, epoch_total_loss, train_loss, train_accuracy, test_loss, test_accuracy))

    print('Finished Training')
    return total_loss_list, training_accuracy_list, testing_accuracy_list

if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    MAX_EPOCH = 5

    baseline_total_loss, baseline_training_accuracy, baseline_testing_accuracy = train_and_evaluate_model(BATCH_SIZE, LEARNING_RATE, MAX_EPOCH)

    add_batch_normalization_total_loss, add_batch_normalization_training_accuracy, add_batch_normalization_testing_accuracy = train_and_evaluate_model(BATCH_SIZE, LEARNING_RATE, MAX_EPOCH, add_batch_normalization=True)
    plot_total_loss(baseline_total_loss, add_batch_normalization_total_loss, MAX_EPOCH, "add_batch_normalization")
    plot_training_accuracy(baseline_training_accuracy, add_batch_normalization_training_accuracy, MAX_EPOCH, "add_batch_normalization")
    plot_testing_accuracy(baseline_testing_accuracy, add_batch_normalization_testing_accuracy, MAX_EPOCH, "add_batch_normalization")

    add_fc_layer_total_loss, add_fc_layer_training_accuracy, add_fc_layer_testing_accuracy = train_and_evaluate_model(BATCH_SIZE, LEARNING_RATE, MAX_EPOCH, add_fc_layer=True)
    plot_total_loss(baseline_total_loss, add_fc_layer_total_loss, MAX_EPOCH, "add_fc_layer")
    plot_training_accuracy(baseline_training_accuracy, add_fc_layer_training_accuracy, MAX_EPOCH, "add_fc_layer")
    plot_testing_accuracy(baseline_testing_accuracy, add_fc_layer_testing_accuracy, MAX_EPOCH, "add_fc_layer")

    reduce_hidden_nodes_total_loss, reduce_hidden_nodes_training_accuracy, reduce_hidden_nodes_testing_accuracy = train_and_evaluate_model(BATCH_SIZE, LEARNING_RATE, MAX_EPOCH, add_fc_layer=True, reduce_hidden_nodes=True)
    plot_total_loss(baseline_total_loss, reduce_hidden_nodes_total_loss, MAX_EPOCH, "reduce_hidden_nodes")
    plot_training_accuracy(baseline_training_accuracy, reduce_hidden_nodes_training_accuracy, MAX_EPOCH, "reduce_hidden_nodes")
    plot_testing_accuracy(baseline_testing_accuracy, reduce_hidden_nodes_testing_accuracy, MAX_EPOCH, "reduce_hidden_nodes")

    halve_num_filters_total_loss, halve_num_filters_training_accuracy, halve_num_filters_testing_accuracy = train_and_evaluate_model(BATCH_SIZE, LEARNING_RATE, MAX_EPOCH, add_fc_layer=True, halve_num_filters=True)
    plot_total_loss(baseline_total_loss, halve_num_filters_total_loss, MAX_EPOCH, "halve_num_filters")
    plot_training_accuracy(baseline_training_accuracy, halve_num_filters_training_accuracy, MAX_EPOCH, "halve_num_filters")
    plot_testing_accuracy(baseline_testing_accuracy, halve_num_filters_testing_accuracy, MAX_EPOCH, "halve_num_filters")

    use_mean_pooling_total_loss, use_mean_pooling_training_accuracy, use_mean_pooling_testing_accuracy = train_and_evaluate_model(BATCH_SIZE, LEARNING_RATE, MAX_EPOCH, add_fc_layer=True, use_mean_pooling=True)
    plot_total_loss(baseline_total_loss, use_mean_pooling_total_loss, MAX_EPOCH, "use_mean_pooling")
    plot_training_accuracy(baseline_training_accuracy, use_mean_pooling_training_accuracy, MAX_EPOCH, "use_mean_pooling")
    plot_testing_accuracy(baseline_testing_accuracy, use_mean_pooling_testing_accuracy, MAX_EPOCH, "use_mean_pooling")

    add_dropout_layer_total_loss, add_dropout_layer_training_accuracy, add_dropout_layer_testing_accuracy = train_and_evaluate_model(BATCH_SIZE, LEARNING_RATE, MAX_EPOCH, add_fc_layer=True, add_dropout_layer=True)
    plot_total_loss(baseline_total_loss, add_dropout_layer_total_loss, MAX_EPOCH, "add_dropout_layer")
    plot_training_accuracy(baseline_training_accuracy, add_dropout_layer_training_accuracy, MAX_EPOCH, "add_dropout_layer")
    plot_testing_accuracy(baseline_testing_accuracy, add_dropout_layer_testing_accuracy, MAX_EPOCH, "add_dropout_layer")
