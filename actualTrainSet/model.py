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
        # return cam, self.labels[index]
        return F.pad(cam, (8 - shape_one, 0, 8-shape_zero, 0)).unsqueeze(0), self.labels[index]
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
        
        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2)
        self.batchnorm4 = nn.BatchNorm2d(64)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)    

        # self.cnn5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2)
        # self.batchnorm5 = nn.BatchNorm2d(128)
        # self.maxpool5 = nn.MaxPool2d(kernel_size=2)   
                
        # self.cnn6 = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=2)
        # self.batchnorm6 = nn.BatchNorm2d(512)
        # self.maxpool6 = nn.MaxPool2d(kernel_size=2)

        # self.cnn7 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=2)
        # self.batchnorm7 = nn.BatchNorm2d(1024)
        # self.maxpool7 = nn.MaxPool2d(kernel_size=2)

        # self.cnn8 = nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=3, stride=1, padding=2)
        # self.batchnorm8 = nn.BatchNorm2d(2048)
        # self.maxpool8 = nn.MaxPool2d(kernel_size=2)


        # self.cnn9 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3, stride=1, padding=2)
        # self.batchnorm9 = nn.BatchNorm2d(1024)
        # self.maxpool9 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(in_features=64, out_features=4000)
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
        # out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)

        out = self.cnn3(out)
        # out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.maxpool3(out)

        out = self.cnn4(out)
        out = self.batchnorm4(out)
        out = self.relu(out)
        out = self.maxpool4(out)

        # out = self.cnn5(out)
        # # out = self.batchnorm5(out)
        # out = self.relu(out)
        # out = self.maxpool5(out)

        # out = self.cnn6(out)
        # # out = self.batchnorm6(out)
        # out = self.relu(out)
        # out = self.maxpool6(out)

        # out = self.cnn7(out)
        # # out = self.batchnorm7(out)
        # out = self.relu(out)
        # out = self.maxpool7(out)

        # out = self.cnn8(out)
        # # out = self.batchnorm8(out)
        # out = self.relu(out)
        # out = self.maxpool8(out)

        # out = self.cnn9(out)
        # out = self.batchnorm9(out)
        # out = self.relu(out)
        # out = self.maxpool9(out)
        #Flattening is done here with .view() -> (batch_size, 32*16*16) = (100, 8192)
        out = out.view(-1,64)   #-1 will automatically update the batchsize as 100; 8192 flattens 32,16,16
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
            print(data.shape)
            input()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)
            # print(train_loss)
            # input()
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
            torch.save(model.state_dict(), 'trainedModel' + str(index) + '.pt')
            valid_loss_min = valid_loss


def test(index):
    model = Net()
    model.load_state_dict(torch.load('trainedModel' + str(index) + '.pt'))
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
        if forRoc[i]>0.8:
            if allPred[i] == allTrue[i]:
                right+=1
            else:
                wrong+=1
    print(right/(wrong+right))
    print(roc_auc_score(allTrue, forRoc))
    print(len([x for x in forRoc if x>0.8]))
    print(len([x for x in forRoc if x<0.5]))

# train(2)
# test(9)

def getEvaluationData():
    testInputs = pickle.load(open('../actualTestSet/' + str(index) + 'testInput', 'rb'))
    testLabels = pickle.load(open('../actualTestSet/' + str(index) + 'testOutput', 'rb'))
    testset = Dataset(testInputs, testLabels)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,shuffle=True, num_workers=24) 
    return testloader

def getEvaluationProbs(index):
    model = Net()
    model.load_state_dict(torch.load('trainedModel' + str(index) + '.pt'))
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.003, momentum= 0.9)
    # for each heatMap in cams2, cams7, cams9 we have to produce a probability
    cams = pickle.load(open('../' + str(index) + 'Cams.pickle', 'rb'))
    print(len(cams))
    input()
    imgToProb = {}
    for img in cams:
        cam = torch.FloatTensor(cams[img])
        shape_zero = len(cam)
        shape_one = len(cam[0])
        tensorIn = F.pad(cam, (8 - shape_one, 0, 8-shape_zero, 0)).unsqueeze(0).unsqueeze(0)
        output = model(tensorIn)
        pred, ind = torch.max(output, 1)
        # print(output)
        # print('prediction: ', pred)
        # print('index: ', ind)
        if ind.data.tolist()[0] == 0:
            prob = 1- pred.data.tolist()[0]
            # print('final prob: ', prob)
        else:
            prob = pred.data.tolist()[0]
            # print('final prob: ', prob)
        imgToProb[img] = prob
        # input()
    # [16,1,8,8]
    pickle.dump(imgToProb, open('imgToProb' + str(index) + '.pickle', 'wb'))
getEvaluationProbs(9)
getEvaluationProbs(7)
getEvaluationProbs(2)

