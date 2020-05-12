import pickle as pkl
f = open('../../claireyin/trainImagesFiltered.txt','r')
imageToTruth = {}
for line in f:
    s = line.split(' ')
    imageToTruth[s[0]] = [int(s[1]), int(s[2]), int(s[3])]
f.close()

with open('trainImageToTruth.pickle', 'wb') as f:
    pkl.dump(imageToTruth, f)