import os
import numpy as np
import time
import sys
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
import re
import pickle
#-------------------------------------------------------------------------------- 
#---- Class to generate heatmaps (CAM)

# pickle_in = open('actualTrainSet/2Cams.pickle', 'rb')
cams2 = {}
# pickle_in2 = open('actualTrainSet/7Cams.pickle', 'rb')
cams7 ={}
# pickle_in3 = open('actualTrainSet/9Cams.pickle', 'rb')
cams9 = {}

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True)
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
          
        model = torch.nn.DataParallel(model)
        modelCheckpoint = torch.load(pathModel, map_location='cpu')
        state_dict = modelCheckpoint['state_dict']
        remove_data_parallel = False # Change if you don't want to use nn.DataParallel(model)
        print('starging this')
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        for key in list(state_dict.keys()):
            match = pattern.match(key)
            new_key = match.group(1) + match.group(2) if match else key
            new_key = new_key[7:] if remove_data_parallel else new_key
            state_dict[new_key] = state_dict[key]
            # Delete old key only if modified.
            if match or remove_data_parallel: 
                del state_dict[key]
        print('done this')
        # if os.path.isfile(CKPT_PATH):
        #     print("=> loading checkpoint")
        #     checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(modelCheckpoint['state_dict'])
        print("=> loaded checkpoint") 
        #addition by claire
        # self.modelP = model
        # self.model = model.module.densenet121.features
        # self.model.eval()
        self.feature_extractor = model.module.densenet121.features
        self.feature_extractor.eval()
        self.classifier = model.module.densenet121.classifier
        
        self.weights = list(self.classifier.parameters())[-2].cpu().data.numpy()
        self.bias = list(self.classifier.parameters())[-1].cpu().data.numpy()

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)

        #claire added for pred
        #-------------------- SETTINGS: DATASET BUILDERS
        # transformList2 = []
        # transResize = 256
        # transformList2.append(transforms.Resize(transResize))
        # transformList2.append(transforms.TenCrop(224))
        # transformList2.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList2.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # self.transformSequence2 = transforms.Compose(transformList2)
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        
        #---- Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        # imageData_pred = self.transformSequence2(imageData) #claire add
        imageData = self.transformSequence(imageData)
        imageData = imageData.unsqueeze_(0)

        #claire add
        # imageData_pred = imageData_pred.unsqueeze_(0)

        # bs, n_crops, c, h, w = imageData_pred.size()
        # varInputPred = torch.autograd.Variable(imageData_pred.view(-1, c, h, w).cuda())
        # self.modelP.cuda()        
        # out_p = self.modelP(varInputPred)
        # outMeanP = out_p.view(bs, n_crops, -1).mean(1)
        # preds2 = outMeanP.data.tolist() #1x14
        # print(preds2)
        #end claire add
        # threeVals = [preds2[0][2], preds2[0][7], preds2[0][9]]
        # decideThree = [x>0.5 for x in threeVals]
        # imageToLabels[pathOutputFile] = decideThree
        
        inputTensor = torch.autograd.Variable(imageData)
        self.feature_extractor
        output = self.feature_extractor(inputTensor)
        output = output.squeeze(0)
        output = output.cpu().data.numpy()
        
        #---- Generate heatmap
        heatmap2= np.zeros((output.shape[-2], output.shape[-1]))
        heatmap7= np.zeros((output.shape[-2], output.shape[-1]))
        heatmap9= np.zeros((output.shape[-2], output.shape[-1]))        
        for j in range(self.weights.shape[1]):
            heatmap2 += self.weights[2][j] * output[j]
            heatmap7 += self.weights[7][j] * output[j]
            heatmap9 += self.weights[9][j] * output[j]

        heatmap2 += self.bias[2]
        heatmap7 += self.bias[7]
        heatmap9 += self.bias[9]
        
        # code below is taken from https://github.com/jrzech/reproduce-chexnet  
        heatmap2=1 / (1 + np.exp(-heatmap2))
        heatmap7=1 / (1 + np.exp(-heatmap7))
        heatmap9=1 / (1 + np.exp(-heatmap9))
        cams2[pathOutputFile] = heatmap2.data.tolist()
        cams7[pathOutputFile] = heatmap7.data.tolist()
        cams9[pathOutputFile] = heatmap9.data.tolist()


        #---- Blend original and heatmap 
        # Heatmap = heatmap2

        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = heatmap2 / np.max(heatmap2)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap2 = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            
        img = heatmap2 * 0.5 + imgOriginal
        #cams[pathOutputFile] = img
        cv2.imwrite('first.jpg', img)
        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = heatmap7 / np.max(heatmap7)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap7 = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            
        img = heatmap7 * 0.5 + imgOriginal
        #cams[pathOutputFile] = img
        cv2.imwrite('second.jpg', img)
        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = heatmap9 / np.max(heatmap9)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap9 = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
            
        img = heatmap9 * 0.5 + imgOriginal
        #cams[pathOutputFile] = img
        cv2.imwrite('third.jpg', img)
        
        print(pathImageFile)
        input()
#-------------------------------------------------------------------------------- 
# alreadyGot = [f for f in os.listdir('heatmaps/')]
pathModel = 'models/m-25012018-123527.pth.tar'

nnArchitecture = 'DENSE-NET-121'
nnClassCount = 14

transCrop = 224

h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
count = 0
with open ('./finalTest.txt', 'r') as f:
    for line in f.readlines():
        if len(line) == 1:
            continue
        pathInputImage = '../data/' + line.split()[0]
        # if pathInputImage in alreadyGot:
        #     continue
        pathOutputImage = line.split()[0].split('/')[4]
        # if pathOutputImage in cams2:
        #     continue
        h.generate(pathInputImage, pathOutputImage, transCrop)
        if count%100 == 0:
            print(count)
        if count%1000 == 0:
            file2 = open('actualTestSet/2Cams.pickle', 'wb')
            file7 = open('actualTestSet/7Cams.pickle', 'wb')
            file9 = open('actualTestSet/9Cams.pickle', 'wb')
            pickle.dump(cams2, file2)
            pickle.dump(cams7, file7)
            pickle.dump(cams9, file9)
        count += 1
file2 = open('actualTestSet/2Cams.pickle', 'wb')
file7 = open('actualTestSet/7Cams.pickle', 'wb')
file9 = open('actualTestSet/9Cams.pickle', 'wb')
pickle.dump(cams2, file2)
pickle.dump(cams7, file7)
pickle.dump(cams9, file9)
