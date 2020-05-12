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
pickle_in = open('cams.pickle', 'rb')
cams = pickle.load(pickle_in)

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, nnArchitecture, nnClassCount, transCrop):
       
        #---- Initialize the network
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, True).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, True).cuda()
          
        model = torch.nn.DataParallel(model).cuda()
        modelCheckpoint = torch.load(pathModel)
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
        self.model2 = model
        self.model = model.module.densenet121.features
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.parameters())[-2]

        #---- Initialize the image transform - resize + normalize
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(transCrop))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        
        self.transformSequence = transforms.Compose(transformList)

        #added from chexnettrainer
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList2 = []
        transResize = 256
        transformList2.append(transforms.Resize(transResize))
        transformList2.append(transforms.TenCrop(224))
        transformList2.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList2.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        self.transformSequence2 = transforms.Compose(transformList2)
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
  
        #---- Load image, transform, convert 
        imageData = Image.open(pathImageFile).convert('RGB')
        imageData2 = self.transformSequence2(imageData2)
        imageData2 = imageData2.unsqueeze_(0)
        imageData = imageData.unsqueeze_(0)
        
        #added from chexnet
        bs, n_crops, c, h, w = imageData2.size()
        varInput2 = torch.autograd.Variable(imageData2.view(-1, c, h, w).cuda())
        self.model2.cuda()        
        out2 = self.model2(varInput2)
        outMean = out.view(bs, n_crops, -1).mean(1)
        predicted = outMean.data.tolist() #16x14
        #print("PREDICTED: " + str(predicted))
        #print("PREDICTED: " + str(outMean.data.size()))
        #end of added from chexnet

        inputTensor = torch.autograd.Variable(imageData)
        self.model.cuda()
        output = self.model(inputTensor.cuda())



        #---- Generate heatmap
        heatmap = None
        for i in range (0, len(self.weights)):
            map = output[0,i,:,:]
            if i == 0: heatmap = self.weights[i] * map
            else: heatmap += self.weights[i] * map
        # print(heatmap.data.tolist())
        # print(type(heatmap.data.tolist()))
        # input()
        cams[pathOutputFile] = heatmap.data.tolist()
        # print(heatmap)
        # print(heatmap.shape)
        # input()
        
        #---- Blend original and heatmap 
        # npHeatmap = heatmap.cpu().data.numpy()

        # imgOriginal = cv2.imread(pathImageFile, 1)
        # imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        # cam = npHeatmap / np.max(npHeatmap)
        # cam = cv2.resize(cam, (transCrop, transCrop))
        # heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
              
        # img = heatmap * 0.5 + imgOriginal
        # # print(type(image))
        # cams[pathOutputFile] = img
        # cv2.imwrite(pathOutputFile, img)
        
#-------------------------------------------------------------------------------- 
alreadyGot = [f for f in os.listdir('heatmaps/')]
pathModel = 'models/m-25012018-123527.pth.tar'

nnArchitecture = 'DENSE-NET-121'
nnClassCount = 14

transCrop = 224

h = HeatmapGenerator(pathModel, nnArchitecture, nnClassCount, transCrop)
count = 0
with open ('../finalTestImages.txt', 'r') as f:
    for line in f.readlines():
        pathInputImage = '../data/' + line.split()[0]
        # if pathInputImage in alreadyGot:
        #     continue
        pathOutputImage = line.split()[0].split('/')[4]
        h.generate(pathInputImage, pathOutputImage, transCrop)
        if count%100 == 0:
            print(count)
        count += 1
        # print(cams)
        # break
        # input()

file = open('cams.pickle', 'wb')
pickle.dump(cams, file)