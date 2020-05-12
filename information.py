import pandas as pd
import os
#chexpert = pd.read_csv('mimic-cxr-2.0.0-chexpert.csv')
#print(len(chexpert))
#posAndNegPleuralEffusion = chexpert[(chexpert['Pleural Effusion'] == 1.0)  | (chexpert['Pleural Effusion'] == 0.0)]
#print(len(posAndNegPleuralEffusion))
#posAndNegPneumothorax  = chexpert[(chexpert['Pneumothorax'] == 1.0)  | (chexpert['Pneumothorax'] == 0.0)]
#print(len(posAndNegPneumothorax))
#
#posAndNegMulti  = chexpert[(chexpert['Pneumothorax'] == 1.0)  | (chexpert['Pneumothorax'] == 0.0)|(chexpert['Pleural Effusion'] == 1.0)  | (chexpert['Pleural Effusion'] == 0.0)|(chexpert['No Finding'] == 1.0)  | (chexpert['No Finding'] == 0.0)| (chexpert['Edema'] == 0.0)|(chexpert['Edema'] == 1.0)].drop(columns=[x for x in chexpert.columns if x not in ['subject_id', 'study_id', 'Edema', 'No Finding', 'Pleural Effusion', 'Pneumothorax']])
#print(len(posAndNegMulti))
#posAndNegMulti.to_csv('usefulStudies.csv')
#print(posAndNegMulti.head(10))
# posAndNegMulti = pd.read_csv('usefulStudies.csv')
# print(posAndNegMulti.head())
# studyIDs = [str(x) for x in posAndNegMulti['study_id']]
# count = 0
# for file in os.listdir('bucket/files/p10'):
#     for f in os.listdir('bucket/files/p10/' + file):
#         if f[1:] in studyIDs:
#             count += 1
#         if count % 50 == 1:
#             print(count)

#imageCount = 0
#for outfile in os.listdir('bucket/files'):
#    print(outfile)
#    for twofile in os.listdir('bucket/files/'+outfile):
#        print(twofile)
#        for threefile in os.listdir('bucket/files/' + outfile + '/' + twofile):
#            if threefile[-3:] == 'jpg':
#                imageCount += 1
#print('total images: ', imageCount)
df  = pd.read_csv('usefulStudies.csv')
justEdema = 0
justPleural = 0
justPneumo = 0
pneumoEdema = 0
pneumoPlueral = 0
edemaPleural = 0
allCount = 0
twoCount = 0
indivCount = 0
noneCount = 0
count = 0
for index, row in df.iterrows():
    count += 1
    total = row.Edema+row.Pleural_Effusion+row.Pneumothorax
    if total == 1.0:
        indivCount += 1
        if row.Edema == 1.0:
            justEdema += 1
        elif row.Pleural_Effusion == 1.0:
            justPleural += 1
        else:
            justPneumo += 1
    elif total == 2.0:
        twoCount += 1
        if row.Edema == 1.0:
            if row.Pleural_Effusion == 1.0:
                edemaPleural += 1
            else:
                pneumoEdema += 1
        elif row.Pleural_Effusion:
            if row.Pneumothorax:
                pneumoPlueral += 1
    elif total == 3.0:
        allCount += 1
    
    elif total == 0.0:
        if row.No_Finding == 1.0:
            noneCount += 1
print(str(justEdema) + ' just have Edema')
print(str(justPleural) + ' just have Pleural Effusion')
print(str(justPneumo) + ' just have Pneumothorax')
print(str(edemaPleural) + ' have edema and Pleural Effusion')
print(str(pneumoPlueral) + ' have Pneumothorax and Pleural Effusion')
print(str(pneumoEdema) + ' have Pneumothorax and Edema')
print(str(allCount) + ' have all three')
print(str(twoCount) + ' have multiple')
print(str(indivCount) + ' have just one')
print(str(noneCount) + ' have none')
print(str(count) + ' total')