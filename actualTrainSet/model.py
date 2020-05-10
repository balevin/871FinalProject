import pickle
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Dataset(data.Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
      #   cam = torch.cuda.FloatTensor(self.inputs[index])
        cam = torch.FloatTensor(self.inputs[index])
        shape_zero = len(cam)
        shape_one = len(cam[0])

        return F.pad(cam, (12 - shape_one, 0, 10-shape_zero, 0)).unsqueeze(0), self.labels[index]
def prepareData(index):
    trainInputs = pickle.load(open(str(index) + 'trainInput', 'rb'))
    trainLabels = pickle.load(open(str(index) + 'trainOutput', 'rb'))
    trainset = Dataset(trainInputs, trainLabels)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,shuffle=True, num_workers=24)
    testInputs = pickle.load(open(str(index) + 'testInput', 'rb'))
    testLabels = pickle.load(open(str(index) + 'testOutput', 'rb'))
    testset = Dataset(testInputs, testLabels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=True, num_workers=24)
    return trainloader, testloader

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)        
        self.relu = nn.ReLU()                 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  
        
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    
        
        self.fc1 = nn.Linear(in_features=192, out_features=4000)
        self.droput = nn.Dropout(p=0.5)                    
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50, out_features=2)    
       
        
    def forward(self,x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
      #   print(out.shape)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(16,192)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
        #Then we forward through our fully connected layer 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc4(out)
        out = self.relu(out)
        out = self.droput(out)
        out = self.fc5(out)
      #   out = self.softmax(self.fc5(out))
        return out


def train(index):
    trainLoader, testLoader = prepareData(index)
    model = Net()
     # model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum= 0.9)
    valid_loss_min = np.Inf
    for epoch in range(10):
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for data, target in trainLoader:
                 # data, target = data.cuda(), target.cuda()
                 # data = data.view(16, 1, 10, 12)
                 # print(data.shape)
            optimizer.zero_grad()
            output = model(data)
                 # print(output.shape)
                 # output = output.reshape(16)
                 # print(output)
                 # print(target)
                 # print(target.shape)
                 # input()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
        model.eval()
        for data, target in testLoader:
                 # data, target = data.cuda(), target.cuda()
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
           
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                 epoch, train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
                 valid_loss))
            torch.save(model.state_dict(), 'trainedModel.pt')
            valid_loss_min = valid_loss

train(2)