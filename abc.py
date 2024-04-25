import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
import os

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the transformations for the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load the data from the train and test directories
train_dir = 'train'
test_dir = 'test'

train_data = []
test_data = []

for class_name in os.listdir(train_dir):
    for file_name in os.listdir(os.path.join(train_dir, class_name)):
        if file_name.endswith('.png'):
            image_path = os.path.join(train_dir, class_name, file_name)
            label = int(class_name)  # assuming the class name is a number
            train_data.append((image_path, label))

for class_name in os.listdir(test_dir):
    for file_name in os.listdir(os.path.join(test_dir, class_name)):
        if file_name.endswith('.png'):
            image_path = os.path.join(test_dir, class_name, file_name)
            label = int(class_name)  # assuming the class name is a number
            test_data.append((image_path, label))

# Split the data into training and testing sets using K-Fold Cross Validation
kfold = KFold(n_splits=5, shuffle=True)

train_labels = [x[1] for x in train_data]
test_labels = [x[1] for x in test_data]

accuracies = []

for train_index, val_index in kfold.split(train_labels):
    X_train, X_val = [], []
    y_train, y_val = [], []

    for i in train_index:
        X_train.append(train_data[i][0])
        y_train.append(train_data[i][1])

    for i in val_index:
        X_val.append(test_data[i][0])
        y_val.append(test_data[i][1])

    # Convert the data into tensors
    X_train = [torch.load(x) for x in X_train]
    X_val = [torch.load(x) for x in X_val]

    # Define the CNN model architecture
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 6, 5)
            self.pool = torch.nn.MaxPool2d(2)
            self.conv2 = torch.nn.Conv2d(6, 16, 5)
            self.fc1 = torch.nn.Linear(16 * 3 * 3, 120)
            self.fc2 = torch.nn.Linear(120, 10)

        def forward(self, x):
            x = torch.relu(torch.flatten(self.pool(torch.relu(self.conv1(x))), start_dim=1))
            x = torch.relu(torch.flatten(self.pool(torch.relu(self.conv2(x))), start_dim=1))
            x = self.fc2(self.fc1(x))
            return x

    # Initialize the model on a specified device (GPU if available, otherwise CPU)
    net = Net().to(device)

    # Define the loss function and optimizer for training the model
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the model on the validation set
    for epoch in range(5):
        running_loss = 0.0
        for i, (images, labels) in enumerate(zip(X_val, y_val)):
            images, labels = images.to(device), torch.tensor(labels).to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {running_loss / (i + 1)}')

    # Evaluate the model on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in zip(X_val, y_val):
            images, labels = images.to(device), torch.tensor(labels).to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2f}')
    accuracies.append(accuracy)

# Print the average test accuracy across all folds
print(f'Average Test Accuracy: {sum(accuracies) / len(accuracies):.2f}')
