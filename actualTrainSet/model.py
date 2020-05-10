import pickle
import torch
from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics.ranking import roc_auc_score

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
    testInputs = pickle.load(open(str(index) + 'valInput', 'rb'))
    testLabels = pickle.load(open(str(index) + 'valOutput', 'rb'))
    testset = Dataset(testInputs, testLabels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=True, num_workers=24)
    return trainloader, testloader

def prepareDataTest(index):
    testInputs = pickle.load(open('../actualTestSet/' + str(index) + 'testInput', 'rb'))
    testLabels = pickle.load(open('../actualTestSet/' + str(index) + 'testOutput', 'rb'))
    testset = Dataset(testInputs, testLabels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=True, num_workers=24) 
    return testloader
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3,stride=1, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(8)        
        self.relu = nn.ReLU()                 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  
        
        self.cnn2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)    
                
        self.cnn3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm3 = nn.BatchNorm2d(32)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)   
        
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)    
                
        self.cnn5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.batchnorm5 = nn.BatchNorm2d(32)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=32, out_features=4000)
        self.droput = nn.Dropout(p=0.5)                    
        self.fc2 = nn.Linear(in_features=4000, out_features=2000)
        self.droput = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(in_features=2000, out_features=500)
        self.droput = nn.Dropout(p=0.5)
        self.fc4 = nn.Linear(in_features=500, out_features=50)
        self.droput = nn.Dropout(p=0.5)
        self.fc5 = nn.Linear(in_features=50, out_features=2)
        self.softmax = nn.Softmax()
       
        
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

        out = self.cnn3(out)
        out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.maxpool3(out)
        #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1,32)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
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
        out = self.softmax(self.fc5(out))
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
           
        train_loss = train_loss/len(trainLoader.dataset)
        valid_loss = valid_loss/len(testLoader.dataset)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                 epoch, train_loss, valid_loss))
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
                 valid_loss))
            torch.save(model.state_dict(), 'trainedModel2.pt')
            valid_loss_min = valid_loss


def test(index):
    model = Net()
    model.load_state_dict(torch.load('trainedModel2.pt'))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum= 0.9)
    testloader = prepareDataTest(index)
    test_loss = 0.0
    class_correct = list(0. for i in range(2))
    class_total = list(0. for i in range(2))

    model.eval()
    i=1
    allPred = []
    allTrue = []
    forRoc = []
    # iterate over test data
    for data, target in testloader:
        output = model(data)
        # target = target.data.tolist()
        # right = 0
        # i = 0
        # print(target[0])
        # for pair in output:
        #     if pair[0]>pair[1]:
        #         if target[i] == 0:
        #             right += 1
        #     if pair[0]<pair[1]:
        #         if target[i] == 1:
        #             right += 1
        # print(right/16)
        
        _, pred = torch.max(output, 1)
        trial = output.data.tolist()
        for i in range(len(pred.data.tolist())):
            if pred.data.tolist()[i] == 1:
                forRoc.append(trial[i][1])
            if pred.data.tolist()[i] == 0:
                forRoc.append(1-trial[i][0])
 
            
        allPred.extend(pred.data.tolist())
        allTrue.extend(target.data.tolist())
        # input()

    right = 0
    wrong = 0
    for i in range(len(allPred)):
        if allPred[i] == allTrue[i]:
            right+=1
        else:
            wrong+=1
    print(right/(wrong+right))
    print(roc_auc_score(allTrue, forRoc))

# train(2)
test(2)