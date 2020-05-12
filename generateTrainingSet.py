import pickle
def makeTrainingSet(index):
    finalInput = []
    finalOutput = []
    cams = pickle.load(open(str(index) + 'Cams.pickle', 'rb')) 
    predictions = pickle.load(open('mageToLabels.pickle', 'rb'))
    truth = pickle.load(open('mageToTruth.pickle', 'rb'))
    if index == 2:
        i = 0
    elif index == 7:
        i = 1
    elif index == 9:
        i=2
    for image, value in cams.items():
        finalInput.append(value)
        pred = predictions[image][i]
        true = truth[image][i]
        if pred == true:
            finalOutput.append(1)
        else:
            finalOutput.append(0)
    pickle.dump(finalInput, open(str(index) + 'testInput', 'wb'))
    pickle.dump(finalOutput, open(str(index) + 'testOutput', 'wb'))


makeTrainingSet(2)
makeTrainingSet(7)
makeTrainingSet(9)