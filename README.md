# 6.871-Final-Project-ErrorNet
## Results Reproduction:
To reproduce the results the data must be mounted as a gcsFuse bucket to a folder titled data. This data was not included here because it is private and extremly large.

## Classification Model
Once the data is aquired, the classification model can be trained and tested by running chexnet/Main.py which uses chexnet/ChexnetTrainer.py

## Heatmaps
To create a heatmaps for each class from an image run heatmap/seperateHeatmapGenerator.py. Information mapping an image name to it's heatmap is stored in 2Cams.pickle, 7Cams.pickle, 9Cams.pickle corresponding to Pleural Effusion, Pneumothorax, and Edema respectively. These heatmaps can be used to generate test and train sets for ErrorNet by running generateTestSet.py and generateTrainSet.py. 

## ErrorNet
Our implementation for this model is stored in actualTrainSet/model.py. Saved state_dict's for each of the three models (one for each category) are stored in actualTrainSet/trainedModel2.pt, actualTrainSet/trainedModel9.pt actualTrainSet/trainedModel9.pt correspoinding to the models for Pleural Effusion, Pneumothorax, and Edema respectively.

## Analysis
Finding optimal thresholds using predicted probabilities for baseline and ErrorNet can be found in evaluate.py. Analyzing these results to see how each performed respectively can be found in analyzeResults.py