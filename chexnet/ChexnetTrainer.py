import os
import numpy as np
import time
import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DatasetGenerator import DatasetGenerator
import re
import pickle

#-------------------------------------------------------------------------------- 
labelList = []
class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        model = torch.nn.DataParallel(model).cuda()
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTrain, transform=transformSequence)
        datasetVal =   DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileVal, transform=transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=24, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=24, pin_memory=True)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')
                
        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)
        
        #---- Load checkpoint 
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])

        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        
        for epochID in range (0, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            lossVal, losstensor = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
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
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.train()
        
        for batchID, (input, target) in enumerate (dataLoader):
                        
            target = target.cuda()
                 
            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(varInput)
            
            lossvalue = loss(varOutput, varTarget)
                       
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0
        with torch.no_grad(): 
            for i, (input, target) in enumerate (dataLoader):
                
                target = target.cuda()
                    
                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)    
                varOutput = model(varInput)
                
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor
                
                lossVal += losstensor.data[0]
                lossValNorm += 1
                
            outLoss = lossVal / lossValNorm
            losstensorMean = losstensorMean / lossValNorm
        
        return outLoss, losstensorMean
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        torch.save(datanpGT, 'groundTruth.pt')
        torch.save(datanpPRED, 'predictions.pt')

        # for i in range(classCount):
        #     try:
        #         outAUROC.append(roc_auc_score(datanpGT[:,i], datanpPRED[:,i], multi_class="ovo", average="macro"))
        #     except Exception as e: 
        #         print(i)
        #         print(e)
        #         print('in making auroc')
        outAUROC.append(roc_auc_score(datanpGT[:,2], datanpPRED[:,2]))
        outAUROC.append(roc_auc_score(datanpGT[:,7], datanpPRED[:,7]))
        outAUROC.append(roc_auc_score(datanpGT[:,9], datanpPRED[:,9]))

        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrainj - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        # print(torch.cuda.get_device_name(0))
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        # cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        # if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained)
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained)
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained)
        
        model = torch.nn.DataParallel(model)
        # model = DenseNet121(N_CLASSES).cuda()
        # model = torch.nn.DataParallel(model).cuda()
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
        # modelCheckpoint = torch.load(pathModel)
        # model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        print('transformed')
        datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=4, shuffle=False, pin_memory=True)
        print('made dataset') 
        # outGT = torch.FloatTensor().cuda()
        # outPRED = torch.FloatTensor().cuda()
        outGT = torch.FloatTensor()
        outPRED = torch.FloatTensor()
        print('pred')
        model.eval()
        print('eval') 
        totalPredictions = []
        totalTruth = []
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):
                if i%5 == 0:
                    print(i)
                # target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = input.size()
                # varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
                varInput = torch.autograd.Variable(input.view(-1, c, h, w))
 
                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                predicted = outMean.data.tolist() #16x14
                truth = target.data.tolist() #16x14
                totalPredictions.extend(predicted)
                totalTruth.extend(truth)

                    # predict_2, predict_7, predict_9 = [], [], []
                    # t2_true, t7_true, t9_true  = [], [], [] 
                    # deferred_2, deferred_7, deferred_9 = 0, 0, 0

                   # for i in range(16):
                       # thisPred = predicted[i]
                       # thisTruth = truth[i]
                       # three = [thisPred[2], thisPred[7], thisPred[9]]
                       # threeTrue = [thisTruth[2], thisTruth[7], thisTruth[9]]
                       #$ decideThree = [x>0.5 for x in three]
                       # labelList.append(decideThree == threeTrue)
                        

                       # if t < three[0] < 1-t:
                       #     deferred_2 += 1
                       # else:
                            #if three[0] >.5:
                            #    predict_2.append(three[0])
                            #    t2_true.append(threeTrue[0])
                            #else:
                                #predict_2.append(1-three[0])
                                #t2_true.append(threeTrue[0])

                   #     if t < three[1] < 1-t:
                   #         deferred_7 += 1
                   #     else:
                   #         if three[1] >.5:
                   #             predict_7.append(three[1])
                   #             t7_true.append(threeTrue[1])
                  #          else:
                  #              predict_7.append(1-three[1])
                  #              t7_true.append(threeTrue[1])
                 #       if t < three[2] < 1-t:
                 #           deferred_9 += 1
                 #       else:
                 #           if three[2] >.5:
                 #               predict_9.append(three[2])
                 #               t9_true.append(threeTrue[2])
                 #           else:
                 #               predict_9.append(1-three[2])
                 #               t9_true.append(threeTrue[2])
                 #   t2.append(predict_2)
                 #   t2_deferred.append(deferred_2)
                 #   t2_label.append(t2_true)
                #=    t7.append(predict_7)
                #    t7_deferred.append(deferred_7)
                #    t7_label.append(t7_true)
                #    t9.append(predict_9)
                #    t9_deferred.append(deferred_9)
                #    t9_label.append(t9_true)
                #t2_deferred = [x/16 for x in t2_deferred]
                #t7_deferred = [x/16 for x in t7_deferred]
                #t9_deferred = [x/16 for x in t9_deferred]

                #t2_auroc = []
                #for i in range(5):
                #    if len(t2_label[i]) == 0:
                #        auc = 0
                #    elif len(t2_label[i]) == 1:
                #        auc = 0
                #    elif (len(set(t2_label[i])) <= 1) == True:
                #        auc = 0
                #    else:
                #        auc = roc_auc_score(t2_label[i], t2[i])
                #    t2_auroc.append(auc)

                #t7_auroc = []
                #for i in range(5):
                #    if len(t7_label[i]) == 0:
                #        auc = 0
                #    elif len(t7_label[i]) == 1:
                #        auc = 0
                #    elif (len(set(t7_label[i])) <= 1) == True:
                #        auc = 0
                #    else:
                #        auc = roc_auc_score(t7_label[i], t7[i])
                #    t7_auroc.append(auc)

                #t9_auroc = []
                #for i in range(5):
                #    if len(t9_label[i]) == 0:
                #        auc = 0
                #    elif len(t9_label[i]) == 1:
                #        auc = 0
                #    elif (len(set(t9_label[i])) <= 1) == True:
                #        auc = 0
                #    else:
                #        auc = roc_auc_score(t9_label[i], t9[i])
                #    t9_auroc.append(auc)


                #with open("t2.txt", "w") as output:
                #    output.write(str(t2))
                #with open("t2_auroc.txt", "w") as output:
                #    output.write(str(t2_auroc))
                #with open("t2_deferred_pct.txt", "w") as output:
                #    output.write(str(t2_deferred))
                #with open("t7.txt", "w") as output:
                    #output.write(str(t7))
                #with open("t7_auroc.txt", "w") as output:
                #    output.write(str(t7_auroc))
                #with open("t7_deferred_pct.txt", "w") as output:
                #    output.write(str(t7_deferred))
           #     with open("t9.txt", "w") as output:
           #         output.write(str(t9))
           #     with open("t9_auroc.txt", "w") as output:
           #         output.write(str(t9_auroc))
           #     with open("t9_deferred_pct.txt", "w") as output:
           #         output.write(str(t9_deferred))    
                
        #        outPRED = torch.cat((outPRED, outMean.data), 0)
                # if i%5 == 0:
                #     print(i)

        #aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        # aurocMean = np.array(aurocIndividual).mean()
        
        # print ('AUROC mean ', aurocMean)
        
        # for i in range (0, len(aurocIndividual)):
        #     try:
        #         print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        #     except Exception as e: 
        #         print(e)
        #         print('that index was not there')
            
        #print('all auroc: ', aurocIndividual)
        # with open('output.txt', 'w') as f:
        #     for i in range() 
        #print('Actual Effusion Score: ', aurocIndividual[0])
       # print('Actual Pneumothorax Score: ', aurocIndividual[1])
       # print('Actual Edema Score: ', aurocIndividual[2])
        with open('predictedProbs.txt', 'w') as f:
            for listy in totalPredictions:
                for item in listy:
                    f.write("%s\n" % item)
        with open('labels.txt', 'w') as f:
            for listy in totalTruth:
                for item in listy:
                    f.write("%s\n" % item)
        return
#-------------------------------------------------------------------------------- 



