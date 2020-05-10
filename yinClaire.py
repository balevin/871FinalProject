import torch
from torch.utils import data
import torch.nn.functional as F
import numpy as np 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image
import cv2
from torchvision import transforms
import pickle as pkl 
import time

#from chexnet.DensenetModels import DenseNet121 as DenseNet

#*****WROTE THESE INITIALIZATIONS TO TAKE IN LABEL 2 ONLY*******

#paths for train set
train_x_path_2 = './chexnet/actualTrainSet/2Cams.pickle'
train_y0_path = './chexnet/actualTrainSet/trainImageToLabels.pickle'
train_y1_path = './chexnet/actualTrainSet/trainImageToTruth.pickle'

#paths for test set
test_x_path_2 = './chexnet/actualTestSet/2Cams.pickle'
test_y0_path = './chexnet/actualTestSet/imageToLabels.pickle'
test_y1_path = './chexnet/actualTestSet/imageToTruth.pickle'

#paths for val set
val_x_path_2 = './chexnet/actualValSet/2Cams.pickle'
val_y0_path = './chexnet/valImageToLabels.pickle'
val_y1_path = './chexnet/valImageToTruth.pickle'

batch_size=64
max_epochs=1000 #max
architecture='DenseNet'
pretrained=False

num_classes=2 # (0,1) binary classification of whether first classification is correct
input_size = [7,8] # heatmap array dimensions

timestampTime = time.strftime("%H%M%S")
timestampDate = time.strftime("%d%m%Y")
timestampSTART = timestampDate + '-' + timestampTime
model_path = '/ErrorNet/ErrorNetTrained'+timestampSTART

loss = torch.nn.BCELoss(reduction = 'mean')

class ErrorNet():
    def __init__ (self, model_path, architecture, pretrained, n_classes):
        """ 
        model_path: path to save model
        architecture: pytorch architecture name
        pretrained: use pretrained model
        n_classes: number of classes in classificatoin
        """
        if architecture=='DenseNet':
            model = DenseNet(n_classes, pretrained)
        self.classes=n_classes
        self.model = torch.nn.DataParallel(model).cuda()
        self.model_path = model_path

    def dataset_generator(self, x_path, y0_path, y1_path):
        """
        Generates data set from pickle files in forms as described below
        Returns np arrays for cams (heatmap array), x (predicted values), and y (true labels)
        cam_path: {img : heatmap array}
        x_path: {img : predictions}
        y_path {img : true labels}
        """
        # Load train set dictionary ({img : heatmap array}, {img : predictions}, {img : true labels})
        with open(x_path, 'rb') as h:
            x_dict = pkl.load(h)
        with open(y0_path, 'rb') as h:
            y0_dict = pkl.load(h)
        with open(y1_path, 'rb') as h:
            y1_dict = pkl.load(h)
        # Sort such that each index corresponds to one image
        x = np.array(self.preprocess([value for (key,value) in sorted(x_dict.items())]))
        y0 = [value for (key,value) in sorted(y0_dict.items())]
        y1= [value for (key,value) in sorted(y1_dict.items())]
        y = np.array([int(y0[i]==y1[i]) for i in range(len(y0))])
        return x, y

    def preprocess(self, x, transCrop=224, transResize=256):
        """
        Preprocess heatmap arrays to be size n by n
        """
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)

        new_x = []
        for heatmap in x:
            cam = (heatmap - np.min(heatmap)) / np.max(heatmap)
            cam = cv2.resize(cam, (transCrop, transCrop))
            new_heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            numpy2pil(new_heatmap)
            new_x.append(transformSequence(numpy2pil(new_heatmap)))
        return new_x

    def train(self, train_x_path, train_y0_path, train_y1_path, val_x_path, val_y0_path, val_y1_path, batch_size, max_epochs, loss):
        """
        train_x_path: path for heatmaps for train set
        train_y0_path: path for initial predicted labels for train set
        train_y1_path: path for true labels for train set
        val_x_path: path for heatmaps for val set
        val_y0_path: path for initial predicted labels for val set
        val_y1_path: path for true labels for val set
        batch_size: train batch size
        max_epochs: train max epochs
        loss: loss settings
        scheduler: scheduler settings
        """
        optimizer = optim.Adam (self.model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
        # Generate and preprocess train set
        train_x, train_y = self.dataset_generator(train_x_path, train_y0_path, train_y1_path)
        datasetTrain = Dataset(train_x, train_y)
        # Create dataloader for train set
        train_loader = data.DataLoader(datasetTrain,
                         batch_size=batch_size,
                         num_workers=24,
                         shuffle=True,
                         pin_memory=True)
        print('made train dataset') 
        # Generate and preprocess val set
        val_x, val_y = self.dataset_generator(val_x_path, val_y0_path, val_y1_path)
        datasetVal = Dataset(val_x, val_y)
        # Create dataloader for val set
        val_loader = data.DataLoader(datasetVal,
                         batch_size=batch_size,
                         num_workers=24,
                         shuffle=False,
                         pin_memory=True)
        print('made val dataset') 
        lossMIN = 100000
        for epochID in range (0, max_epochs):
            self.epoch_train (train_loader, optimizer, loss)
            lossVal, losstensor = self.epoch_val(valloader, optimizer, loss)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.data[0])

            if lossVal < lossMIN:
                lossMIN = lossVal    
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                     
    def epoch_train (self, dataLoader, optimizer, loss):
        """
        Helper function for training per epoch
        """
        self.model.train()
        for batchID, (input, target) in enumerate (dataLoader):   
            target = target.cuda()
                 
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = self.model(varInput)
            
            lossvalue = loss(varOutput, varTarget)
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
                    
    def epochVal (self, dataLoader, optimizer, loss):
        """
        Helper function for val per epoch
        """
        self.model.eval()
        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0
        with torch.no_grad(): 
            for i, (input, target) in enumerate (dataLoader):
                target = target.cuda()
        
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)    
                varOutput = self.model(varInput)
                
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor
                
                lossVal += losstensor.data[0]
                lossValNorm += 1
                
            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm
        return outLoss, losstensorMean
            
    def test(self, test_x_path, test_y0_path, test_y1_path):
        """
        test_x_path: path for heatmap for test set
        test_y0_path: path for initial predicted labels for test set
        test_y1_path: path for true labels for test set
        """
        print(torch.cuda.get_device_name(0))
        cudnn.benchmark = True
        test_x, test_y = self.dataset_generator(test_x_path, test_y0_path, test_y1_path)
        datasetTest = Dataset(test_x, test_y)
        test_loader = data.DataLoader(datasetTest,
                         batch_size=batch_size,
                         num_workers=24,
                         shuffle=False)
        print('made test dataset')

        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        print('pred')
        model.eval()
        print('eval') 

        labelList=[]
        # start
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                if i%5 == 0:
                    print(i)
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = input.size()
                varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
                
                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                predicted = outMean.data.tolist()
                truth = target.data.tolist()

                for i in range(16):
                    p = predicted[i]
                    t = truth[i]
                    labelList.append(int(p>0.5) == t)
                
                outPRED = torch.cat((outPRED, outMean.data), 0)
        # end 
        auroc = self.computeAUROC(outGT, outPRED)
        print('all auroc: ', auroc)

        
        def computeAUROC(self, outGT, outPRED):
            outAUROC = []
        
            datanpGT = dataGT.cpu().numpy()
            datanpPRED = dataPRED.cpu().numpy()
            torch.save(datanpGT, 'groundTruth.pt')
            torch.save(datanpPRED, 'predictions.pt')

            outAUROC.append(roc_auc_score(datanpGT, datanpPRED))

            return outAUROC

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
        cam = self.inputs[index]
        shape_zero = len(cam)
        shape_one = len(cam[0])

        return cam, self.labels[index]

def DenseNet(pretrained=False, progress=True, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """
    return torchvision.models.densenet._densenet('densenet121', 32, (6, 12, 24, 16), 64, pretrained, progress,
                     **kwargs)

def numpy2pil(np_array: np.ndarray) -> Image:
    """
    Convert an HxWx3 numpy array into an RGB Image
    """

    assert_msg = 'Input shall be a HxWx3 ndarray'
    assert isinstance(np_array, np.ndarray), assert_msg
    assert len(np_array.shape) == 3, assert_msg
    assert np_array.shape[2] == 3, assert_msg

    img = Image.fromarray(np_array, 'RGB')
    return img

def main():
    
    model2 = ErrorNet(model_path, architecture, pretrained, num_classes)
    model2.train(train_x_path_2, train_y0_path, train_y1_path, test_x_path_2, test_y0_path, test_y1_path, batch_size, max_epochs, loss)
    #model2.test(test_x_path_2, test_y0_path, test_y1_path)



if __name__ == '__main__':
    main()
    