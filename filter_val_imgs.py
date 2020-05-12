import pandas as pd

val_img_path = '../bradlevin/valImages.txt'
new_val_path = 'valImagesFiltered.txt'
#read lines 
f = open(val_img_path, 'r')
new_f = open(new_val_path, 'w')

for line in f:
    if line.split(' ')[1] != '-1' and line.split(' ')[2] != '-1' and line.split(' ')[3] != '-1':
        new_f.write(line)


f.close()
new_f.close()