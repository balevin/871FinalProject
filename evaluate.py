from sklearn.metrics.ranking import roc_auc_score
# preds = [float(x[:-1]) for x in open('trialPreds.txt', 'r').readlines()]
# labels = [int(x) for x in open('trialLabels.txt', 'r').readlines()]
import numpy as np
import random
import pickle
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
        labels = [int(x) for x in pickle.load(open('data/lab' + cat + 's', 'rb'))]
        for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]:
            predicted = []
            predictedLabels = []
            defferedCount = 0
            for i in range(len(preds)):
                if preds[i]<=threshold or preds[i]>=(1-threshold):
                    predicted.append(preds[i])
                    predictedLabels.append(labels[i])
                else:
                    defferedCount += 1
            # print(predictedLabels)
            # print(predicted)
            score = roc_auc_score(predictedLabels, predicted)
            with open('results/' + str(cat) + "_" + str(threshold) + "_" + 'result.txt', 'w') as f:
                f.write(str(score))
                f.write('\n')
                f.write(str(defferedCount))


# images = [x for ]

# def ourScore():


baselineScore()