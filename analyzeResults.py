import os

def getBest(category, baseline=False):
# AUROC: 0.7258742621672791
# Predicted #: 1834
# Deferred #: 47
# Predicted percentage #: 0.9750132908027644
# Score: 0.7077370530647474
    if baseline:
        path = 'results/'
    else:
        path = 'ourResults/'
    bestAUROC = 0
    bestAUROC_T = 0
    highestPercentage = 0
    highestPercentage_T = 0
    bestScore = 0
    bestScore_T = 0
    for file in os.listdir(path):
        if file[0] != str(category):
            continue
        i = 0
        for line in open(path+file, 'rb').readlines():
            if i == 0:
                if float(line.split()[1])>bestAUROC:
                   bestAUROC = float(line.split()[1])
                   bestAUROC_T = file.split('_')[1]
            if i == 3:
                if float(line.split()[3])>highestPercentage:
                    highestPercentage = float(line.split()[3]) 
                    highestPercentage_T = file.split('_')[1]
            if i == 4:
                if float(line.split()[1])>bestScore:
                    bestScore = float(line.split()[1])
                    bestScore_T = file.split('_')[1]
            i+=1
    if baseline:
        outpath = 'baselineBest/'
    else:
        outpath = 'ourBest/'

    with open(outpath + str(category) + '.txt', 'w') as f:
        f.write('Best AUROC: ' + str(bestAUROC) + ' happens when t=' + str(bestAUROC_T))
        f.write("\n")
        f.write("Highest percentage: " + str(highestPercentage) + ' happens when t=' + str(highestPercentage_T))
        f.write("\n")
        f.write("Best Score: " + str(bestScore) + ' happens when t=' + str(bestScore_T))


def getEmAll():
    for baseline in [True, False]:
        for category in (2,7,9):
            getBest(category, baseline)

getEmAll()