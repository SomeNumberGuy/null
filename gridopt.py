import torch
import torch.nn as nn
import torch.optim as optim
from itertools import product

# Define the neural network model
class Net(nn.Module):
    def __init__(self, num_frozen_layers):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)  # input layer (28x28 images) -> hidden layer (128 units)
        self.fc2 = nn.Linear(128, 10)  # hidden layer (128 units) -> output layer (10 units)

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # activation function for hidden layer
        x = self.fc2(x)
        return x

# Define the optimization hyperparameters to search over
batch_sizes = [32, 64, 128]
num_frozen_layers_options = [0, 1, 2]  # freeze no layers, or up to 2 layers
learning_rates = [0.01, 0.001, 0.0001]
adam_beta1s = [0.9, 0.8, 0.7]
adam_beta2s = [0.99, 0.999, 0.9999]

# Perform the grid search
best_model = None
best_accuracy = 0.0

for batch_size, num_frozen_layers, learning_rate, adam_beta1, adam_beta2 in product(batch_sizes, num_frozen_layers_options, learning_rates, adam_beta1s, adam_beta2s):
    # Freeze the specified number of layers
    net = Net(num_frozen_layers)
    for layer_idx in range(num_frozen_layers):
        if layer_idx < 0:
            continue
        net.fc1.weight.requires_grad = False

    # Set the Adam optimization algorithm parameters
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))

    # Train the model using the specified batch size
    for epoch in range(5):
        running_loss = 0.0
        for i, batch in enumerate(data_loader):
            inputs, labels = batch
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() / len(batch)

    # Evaluate the model's accuracy on a validation set
    accuracy = evaluate_model(net, data_loader)

    if accuracy > best_accuracy:
        best_model = net
        best_accuracy = accuracy

# Print the results
print("Best model:", best_model)
print("Best accuracy:", best_accuracy)
