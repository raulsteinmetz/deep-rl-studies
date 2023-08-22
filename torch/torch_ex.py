import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T

class LinearClassifier(nn.Module):
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(*input_dims, 128) # *input_dims is to unpack lists or tuples to a number
        self.fc2 = nn.Linear(128, 256) # input = 128 units, output = 256 units
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) 
        # self.parameters are the parameters from nn.Module, tell the optimizer what we want to optimize

        self.loss = nn.CrossEntropyLoss() # loss function for classification problems


        # device is where we are sending the calculations to
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data)) # activation for linear classifiers
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2) # cross entropy loss will handle activation

        return layer3
    
    def learn(self, data, labels):
        self.optimizer.zero_grad()
        # optimizer keeps the gradient calculation from last loop, we dp not want that, so we zero it

        # converting np.arrays from data and labels to tensors that will go to gpu
        data = T.tensor(data).to(self.device)
        labels = T.tensor(labels).to(self.device)
        
        predictions = self.forward(data) # forward

        cost = self.loss(predictions, labels) # calculating error

        cost.backward() # backpropagation
        self.optimizer.step()
