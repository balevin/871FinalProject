from sklearn.metrics.ranking import roc_auc_score
# preds = [float(x[:-1]) for x in open('trialPreds.txt', 'r').readlines()]
# labels = [int(x) for x in open('trialLabels.txt', 'r').readlines()]
import numpy as np
import random
import pickle
# from model import *
a = []
# for i in range(100):
#     ap = []
#     for b in range(14):
#         c = random.random()
#         ap.append(c)
#         # if c>.5:
#         #     ap.append(1)
#         # else:
#         #     ap.append(0)
#     a.append(ap)
# print(len(a))
# print(len(a[0]))
# input()
# with open('predictedProbs.txt', 'w') as f:
#     for listy in a:
#         for item in listy:
#             f.write("%s\n" % item)
# input()
# print(a)
# pred2s = []
# pred7s = []
# pred9s = []
# lab2s = []
# lab7s = []
# lab9s = []
# allPred = [x for x in open('predictedProbs.txt', 'r').readlines()]
# allLab = [x for x in open('predictedLabels.txt', 'r').readlines()]
# for i in range(len(allPred)):
#     if i%14==2:
#         pred2s.append(allPred[i])
#         lab2s.append(allLab[i])
#     if i%14==7:
#         pred7s.append(allPred[i])
#         lab7s.append(allLab[i])
#     if i%14 == 9:
#         pred9s.append(allPred[i])
#         lab9s.append(allLab[i])
# pickle.dump(pred2s,open('data/pred2s', 'wb'))
# pickle.dump(pred7s,open('data/pred7s', 'wb'))
# pickle.dump(pred9s,open('data/pred9s', 'wb'))
# pickle.dump(lab2s, open('data/lab2s', 'wb'))
# pickle.dump(lab7s, open('data/lab7s', 'wb'))
# pickle.dump(lab9s, open('data/lab9s', 'wb'))

def baselineScore():
    for cat in ['2', '7', '9']:
        preds = [float(x) for x in pickle.load(open('data/pred' + cat + 's', 'rb'))]
        labels = [int(float(x)) for x in pickle.load(open('data/lab' + cat + 's', 'rb'))]
        print(len(preds))
        input()
        d = {}
        for place in labels:
            if place not in d:
                d[place] = 1
            else:
                d[place] += 1
        print(d)
        input
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            predicted = []
            predictedLabels = []
            defferedCount = 0
            predictedCount = 0
            for i in range(len(preds)):
                if labels[i] == -1:
                    continue
                if preds[i]<=threshold or preds[i]>=(1-threshold):
                    predicted.append(preds[i])
                    predictedLabels.append(labels[i])
                    predictedCount += 1
                else:
                    defferedCount += 1
            # print(predictedLabels)
            # print(predicted)
            score = roc_auc_score(predictedLabels, predicted)
            with open('results/' + str(cat) + "_" + str(threshold) + "_" + 'result.txt', 'w') as f:
                f.write('AUROC: ' + str(score))
                f.write('\n')
                f.write('Predicted #: ' + str(predictedCount))
                f.write('\n')
                f.write('Deferred #: '+ str(defferedCount))
                f.write('\n')
                f.write('Predicted percentage #: ' + str(predictedCount/(predictedCount+defferedCount)))
                f.write('\n')
                f.write('Score: ' + str(score*(predictedCount/(predictedCount+defferedCount))))
                


# images = [x for ]

# def ourScore():


# baselineScore()
# 
# for computing our model, and not running heatmap:
#   download test cams (img:heatmap)
# use testImages to go through each image, get heatmap from dict
# yoss heatmap into our model, get prediction
cam2 = pickle.load(open('2Cams.pickle', 'rb'))
cam7 = pickle.load(open('7Cams.pickle', 'rb'))
cam9 = pickle.load(open('9Cams.pickle', 'rb'))
def camScore():
    for cat in ['2', '7', '9']:
        preds = [float(x) for x in pickle.load(open('data/pred' + cat + 's', 'rb'))]
        labels = [int(float(x)) for x in pickle.load(open('data/lab' + cat + 's', 'rb'))]
        probs = [float(x) for x in pickle.load(open('probabilies' + cat, 'rb'))] 
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            predicted = []
            predictedLabels = []
            defferedCount = 0
            predictedCount = 0
            for i in range(len(preds)):
                if labels[i] == -1:
                    continue
                if probs[i]<=threshold or probs[i]>=(1-threshold):
                    predicted.append(preds[i])
                    predictedLabels.append(labels[i])
                    predictedCount += 1
                else:
                    defferedCount += 1
            # print(predictedLabels)
            # print(predicted)
            score = roc_auc_score(predictedLabels, predicted)
            with open('ourResults/' + str(cat) + "_" + str(threshold) + "_" + 'result.txt', 'w') as f:
                f.write('AUROC: ' + str(score))
                f.write('\n')
                f.write('Predicted #: ' + str(predictedCount))
                f.write('\n')
                f.write('Deferred #: '+ str(defferedCount))
                f.write('\n')
                f.write('Predicted percentage #: ' + str(predictedCount/(predictedCount+defferedCount)))
                f.write('\n')
                f.write('Score: ' + str(score*(predictedCount/(predictedCount+defferedCount))))


camScore()
# imgToProb2 = pickle.load(open('actualTrainSet/imgToProb2.pickle', 'rb'))
# imgToProb7 = pickle.load(open('actualTrainSet/imgToProb7.pickle', 'rb'))
# imgToProb9 = pickle.load(open('actualTrainSet/imgToProb9.pickle', 'rb'))
# print(len(imgToProb2))
# ourPreds2 = []
# ourPreds7 = []
# ourPreds9 = []
# count = 0
# with open('testImages.txt') as f:
#     for file in f.readlines():
#         img = file.split()[0].split('/')[4]
#         ourPreds2.append(imgToProb2[img])
#         ourPreds7.append(imgToProb7[img])
#         ourPreds9.append(imgToProb9[img])

# pickle.dump(ourPreds2, open('probabilies2', 'wb'))
# pickle.dump(ourPreds7, open('probabilies7', 'wb'))
# pickle.dump(ourPreds9, open('probabilies9', 'wb'))
