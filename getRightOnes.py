import pandas as pd
from pathToLabels import getLabelFromPath

allStudies = pd.read_csv('full_frontal.csv')
studies = allStudies['study_id']
meta = pd.read_csv('mimic-cxr-2.0.0-metadata.csv')
meta = meta[meta['study_id'].isin(studies)]
totalImgList = []
for index, row in meta.iterrows():
    if row.ViewPosition == 'AP':
        totalImgList.append(row.dicom_id)

split = pd.read_csv('mimic-cxr-2.0.0-split.csv')
trainImages = []
testImages = []
validateImages = []
for index, row in split.iterrows():
    if row.dicom_id in totalImgList:
        path = 'files/p' + str(row.subject_id)[0:2] + '/' + 'p' + str(row.subject_id) + '/s' + str(row.study_id) + '/' + row.dicom_id + '.jpg'
        labels = getLabelFromPath(path)
        print(path)
        print(labels)
        input()
        if row.split == 'train':
            trainImages.append(path)
        elif row.split == 'test':
            testImages.append(path)
        elif row.split == 'validate':
            validateImages.append(path)


with open('trainImages.txt', 'w') as f:
    for item in trainImages:
        f.write("%s\n" % item)

with open('testImages.txt', 'w') as f:
    for item in testImages:
        f.write("%s\n" % item)

with open('valImages.txt', 'w') as f:
    for item in validateImages:
        f.write("%s\n" % item)