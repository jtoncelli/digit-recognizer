import torch
import torch.nn as nn

class NeuralNetDigitIdentifier(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output):
        super(NeuralNetDigitIdentifier, self).__init__()
        self.l1 = nn.Linear(input_size, hidden1) 
        self.l2 = nn.Linear(hidden1, hidden2)
        self.l3 = nn.Linear(hidden2, output)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        
        out = self.l2(out)
        out = self.relu(out)

        out = self.l3(out)

        return out