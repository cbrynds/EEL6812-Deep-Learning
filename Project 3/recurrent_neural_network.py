import random
import itertools
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch
import matplotlib.pyplot as plt
import matplotlib
import torch.nn as nn
import torch.nn.functional as F
from plotting_utils import (
    plot_test_accuracy,
    plot_test_loss,
    plot_train_accuracy,
    plot_train_loss,
)
import torchinfo

font = {'weight' : 'normal','size'   : 22}
matplotlib.rc('font', **font)
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

from Reber import ReberDataset

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = torch.LongTensor([len(x) for x in xx])
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy = torch.LongTensor(yy)
  return xx_pad, yy, x_lens


def train_model(model, train_loader, test_loader, device, epochs=1000, lr=0.001):

    # Add Loss function to be used 
    # I chose cross entropy loss, which is used commonly for classification tasks
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0.00001)


    plot_data = {"train":[], "test":[]}
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        correct = 0
        for j, (x, y, l) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x, l)
            
            # Compute loss, do backpropagation, and update the paramaters
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.max(y_pred, 1)[1]
            correct += (pred == y).float().sum().item()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
            avg_len = torch.mean(torch.tensor(l, dtype=float)).item()
            
        train_loss = sum_loss / total
        train_acc = correct / total
        plot_data['train'].append((i+1, train_loss, train_acc))

        test_loss, test_acc = test_metrics(model, test_loader, device)


        logging.warning("[Epoch {:3}]   Loss:  {:8.4}     Train Acc:  {:8.4}%     Test Acc:  {:8.4}%".format(i+1, train_loss, train_acc*100, test_acc*100))

        plot_data['test'].append((i+1, test_loss, test_acc))
    return plot_data


def summarize_and_print_metrics(model_name, plot_data):
    train_epochs = plot_data["train"]
    test_epochs = plot_data["test"]

    final_train_epoch, final_train_loss, final_train_acc = train_epochs[-1]
    final_test_epoch, final_test_loss, final_test_acc = test_epochs[-1]

    min_train_loss_epoch, min_train_loss, _ = min(train_epochs, key=lambda x: x[1])
    max_train_acc_epoch, _, max_train_acc = max(train_epochs, key=lambda x: x[2])
    min_test_loss_epoch, min_test_loss, _ = min(test_epochs, key=lambda x: x[1])
    max_test_acc_epoch, _, max_test_acc = max(test_epochs, key=lambda x: x[2])

    print(f"\n{model_name} Metrics Summary:")
    print(
        f"Final Training Loss: {final_train_loss:.4f} | "
        f"Final Training Accuracy: {final_train_acc * 100:.2f}%"
    )
    print(
        f"Final Testing Loss:  {final_test_loss:.4f} | "
        f"Final Testing Accuracy:  {final_test_acc * 100:.2f}%"
    )
    print(
        f"Minimum Training Loss: {min_train_loss:.4f} at epoch {min_train_loss_epoch}"
    )
    print(
        f"Maximum Training Accuracy: {max_train_acc * 100:.2f}% at epoch {max_train_acc_epoch}"
    )
    print(
        f"Minimum Testing Loss: {min_test_loss:.4f} at epoch {min_test_loss_epoch}"
    )
    print(
        f"Maximum Testing Accuracy: {max_test_acc * 100:.2f}% at epoch {max_test_acc_epoch}"
    )

def test_metrics (model, loader, device):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_ae = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for i, (x, y, l) in enumerate(loader):
            x = x.to(device)
            y= y.to(device)

            y_hat = model(x, l)
            loss = criterion(y_hat, y)
            pred = torch.max(y_hat, 1)[1]
            correct += (pred == y).float().sum().item()
            total += y.shape[0]
            sum_loss += loss*y.shape[0]
    return (sum_loss/total).item(), (correct/total)




class SimpleRecurrentClassifier(torch.nn.Module) :
    def __init__(self, hidden_dim=64, embedding_dim=32, recurrent_type=nn.RNN) :
        super().__init__()

        # An embedding layer learns a vector for each of the input symbols
        # Converts BxT_max to BxT_max, embedding_dim by replacing each
        # symbol ID with its corresponding learned vector.
        self.embed = nn.Embedding(8, embedding_dim)

        # A recurrent neural network of the specified type
        self.rnn = recurrent_type(embedding_dim, hidden_dim, batch_first=True, num_layers=1)

        # A single linear layer to convert from RNN hidden state two-class probabilities
        self.classifier = nn.Linear(hidden_dim, 2)
        
    
    def forward(self, x, s):
        
        # Convert symbols IDs of the sequence to embedding vectors
        x = self.embed(x)

        # First, we pack using pack_padded_sequence so that we can efficiently process batches of variable length
        packed_x = nn.utils.rnn.pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)

        # Pass them to recurrent network
        packed_output, _ = self.rnn(packed_x)

        # Unpack the output
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Take final hidden state for each sequence (use sequence
        # lengths to know when each stopped)
        batch_idx = torch.arange(output.size(0), device=output.device)
        s_device = s.to(output.device)
        x = output[batch_idx, s_device-1, :]

        # Pass these through classifier
        x = self.classifier(x)
        return x

def main():
    B=32  # Batch Size
    EmbeddedRepeat = 5 # Number of Reber Grammar Strings between Prefix/Suffix in ERG
    train = ReberDataset(split="train", size=6000, repeat=EmbeddedRepeat)
    test = ReberDataset(split="test", size=2000, repeat=EmbeddedRepeat)
    train_loader = DataLoader(train, batch_size=B, shuffle=True, collate_fn=pad_collate)
    test_loader = DataLoader(test, batch_size=500, shuffle=False, collate_fn=pad_collate)

    logging.warning(["ERG Repeat: {}  Avg.Len: {}".format(EmbeddedRepeat, train.avgLen())])


    ###########################################################################
    # Modify the model definition here to swap out recurrent network types
    ###########################################################################
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
  
    model = SimpleRecurrentClassifier(recurrent_type=nn.LSTM) #nn.RNN, nn.LSTM, or nn.GRU are options
    model.to(device)
    print("LSTM Model Summary:")
    print(torchinfo.summary(model))
    print(model)

    lstm_plot_data = train_model(model, train_loader, test_loader, device)

    model = SimpleRecurrentClassifier(recurrent_type=nn.RNN)
    model.to(device)
    print("RNN Model Summary:")
    print(torchinfo.summary(model))
    print(model)

    rnn_plot_data = train_model(model, train_loader, test_loader, device)

    model = SimpleRecurrentClassifier(recurrent_type=nn.GRU)
    model.to(device)
    print("GRU Model Summary:")
    print(torchinfo.summary(model))
    print(model)

    gru_plot_data = train_model(model, train_loader, test_loader, device)

    # Plot Train Loss, Test Loss, Train accuracy, Test accuracy
    plot_train_loss(lstm_plot_data, rnn_plot_data, gru_plot_data)
    plot_test_loss(lstm_plot_data, rnn_plot_data, gru_plot_data)
    plot_train_accuracy(lstm_plot_data, rnn_plot_data, gru_plot_data)
    plot_test_accuracy(lstm_plot_data, rnn_plot_data, gru_plot_data)

    summarize_and_print_metrics("LSTM", lstm_plot_data)
    summarize_and_print_metrics("RNN", rnn_plot_data)
    summarize_and_print_metrics("GRU", gru_plot_data)
    
if __name__ == "__main__":
    main()

