import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import torch.nn.functional as F

from data import DigitsDataset
from neuralnet import NeuralNetDigitIdentifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 784 
hidden_size1 = 128
hidden_size2 = 64
num_classes = 10  
num_epochs = 15
batch_size = 128
learning_rate = 0.003  

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
test_dataset = DigitsDataset(dataset_path="test.csv", dataset_type="test")

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = NeuralNetDigitIdentifier(input_size=input_size, hidden1=hidden_size1, hidden2=hidden_size2, output=num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device) 

        outputs = model(images)

        loss = criterion(outputs, labels) 
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 

        if i % 100 == 0:
            print(f'epoch {epoch} / {num_epochs}, step {i}/{n_total_steps}, loss = {loss.item():.4f}')


    n_correct = 0
    n_samples = 0
    for images, labels in val_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()

    acc = 100.0 * n_correct / n_samples  
    print(f'accuracy = {acc} at epoch: {epoch}')

predictions = []
with torch.no_grad():
    for data in test_loader:
        outputs = model(data)
        probabilities = F.softmax(outputs, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        predictions.append(predicted_labels)

predictions = torch.cat(predictions, dim=0).tolist()
results_df = pd.DataFrame({'ImageID': range(1, len(predictions)+1), 'Label': predictions})
results_df.to_csv('predictions.csv', index=False)